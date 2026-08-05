import os
from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Tuple, Union

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(str(os.getcwd())).resolve()))

from utils.logger import logger
from utils.utils import *
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from src.vlnce_src.closeloop_util_nohelp import (
    EvalBatchState,
    BatchIterator,
    setup,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
)

from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# Config
# ============================================================

K_FUTURE = 5
USE_XYZW = True


# Corridor
CORRIDOR_BOUNDS = {
    "front_left": (-45.0, -15.0),
    "front": (-15.0, 15.0),
    "front_right": (15.0, 45.0),
}

CORRIDOR_DIST_EPS = 0.5
CORRIDOR_TOLERANCE = 0.0

# ============================================================
# Quaternion / coordinate transform
# ============================================================

def quat_to_rotmat(q: List[float], use_xyzw: bool = True) -> np.ndarray:
    """
    Quaternion -> rotation matrix.

    If use_xyzw=True:
        q = [x, y, z, w]
    Else:
        q = [w, x, y, z]

    Return:
        R: body frame -> world frame
    """
    if use_xyzw:
        x, y, z, w = q
    else:
        w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def world_to_body(
    delta_world: np.ndarray,
    q_curr: List[float],
    use_xyzw: bool = True,
) -> np.ndarray:
    """
    Convert world-frame displacement into UAV body frame.

    If R is body -> world, then:
        body = R.T @ world
    """
    R_mat = quat_to_rotmat(q_curr, use_xyzw=use_xyzw)
    return R_mat.T @ delta_world



SYSTEM_PROMPT = "You are an intelligent UAV navigation agent with spatial geometric reasoning ability. You will be given two images in fixed order: (1) Current Front View, (2) Current Downward View. The UAV body frame is defined as: x forward, y right, z down. The current corridor state is computed from UAV odometry and should be respected as a soft geometric condition. You should generate a corridor-consistent navigation decision and five future waypoints in the current UAV body frame."


def initial_direction_to_corridor(init_direction: str) -> Union[List[float], str]:
    """
    Convert initial direction prior into initial flight corridor.
    """
    if init_direction == "down":
        return "down"

    return list(CORRIDOR_BOUNDS.get(init_direction, (-15.0, 15.0)))


def corridor_to_prompt_text(corridor: Union[List[float], str]) -> str:
    if corridor == "down":
        return "downward-view terminal corridor"

    return f"[{float(corridor[0]):.1f}°, {float(corridor[1]):.1f}°] in the initial UAV body frame"


def build_user_prompt(
    instruction: str,
    init_direction: str,
    corridor: Union[List[float], str],
    current_corridor_state: str,
) -> str:
    """
    This follows the same style as the training-data construction script.
    """

    return (
        "<image>\n<image>\n"
        f"Instruction: {instruction}\n"
        f"Initial direction prior: {init_direction}.\n"
        f"Current corridor state: {current_corridor_state}.\n"
        "Return a JSON object with keys: "
        "\"target_direction\", \"diagnosis\", \"action\", \"stop\", \"waypoints_body\".\n"
        "Constraints:\n"
        "1. target_direction describes the target's current coarse region relative to the UAV body frame. "
        "It must be one of: front_left, front, front_right, down.\n"
        "2. current_corridor_state is an input condition and is one of: "
        "corridor_centered, corridor_left_deviation, corridor_right_deviation, corridor_down_approach.\n"
        "3. diagnosis must be one of: forward_progress, lateral_realign_left, lateral_realign_right, "
        "vertical_realign, approach_and_contract, terminal_converging, fine_realign.\n"
        "4. action must be one of: forward, forward_left, forward_right, forward_down, descend, "
        "align_left_down, align_right_down, hover.\n"
        "5. action must be geometrically consistent with current_corridor_state, target_direction, and waypoints_body.\n"
        "6. stop must be yes only when the UAV is close to the target and the motion is stably converging; otherwise no.\n"
        "7. waypoints_body must contain exactly 5 future waypoints in the current UAV body frame.\n"
        "8. Output valid JSON only."
    )


# ============================================================
# Direction parsing without target_pos
# ============================================================

def parse_initial_direction_from_instruction(text: str) -> Union[str, None]:
    """
    Parse initial coarse direction from the instruction text.

    Return:
        front_left, front, front_right, down, or None

    This does NOT use target_pos.
    """
    # t = text.lower()
    # print("t--------------------------------",t)
    t = text.strip()
    # Downward cases
    if "It's down below you" in t:
        return "down"

    # Front-left cases
    if "It's off to the front-left" in t:
        return "front_left"

    # Front-right cases
    if "It's off to the front-right" in t:
        return "front_right"

    # Front cases
    if "It's straight ahead" in t:
        return "front"

    return None


def strip_direction_sentence(raw_instruction: str) -> str:
    """
    Keep target description while removing coarse direction sentence if present.

    Your previous code used:
        parts = description.split("It's")
        description = parts[0].strip()

    This function keeps the same behavior but is safer.
    """
    if "It's" in raw_instruction:
        return raw_instruction.split("It's")[0].strip()

    if "It is" in raw_instruction:
        return raw_instruction.split("It is")[0].strip()

    return raw_instruction.strip()

def point_angle_in_initial_body(
    p_init: List[float],
    q_init: List[float],
    p_world: List[float],
    use_xyzw: bool = True
) -> Tuple[float, float, List[float]]:
    """
    Compute current position angle relative to initial UAV body frame.
    """
    delta_world = np.array(p_world, dtype=np.float32) - np.array(p_init, dtype=np.float32)
    delta_init_body = world_to_body(delta_world, q_init, use_xyzw=use_xyzw)

    x = float(delta_init_body[0])
    y = float(delta_init_body[1])
    dist_xy = float(np.linalg.norm(delta_init_body[:2]))
    angle = float(np.degrees(np.arctan2(y, max(x, 1e-6))))

    return angle, dist_xy, delta_init_body.tolist()


# ============================================================
# Online corridor state without target_pos
# ============================================================

def infer_current_corridor_state_online_no_target(
    init_direction: str,
    init_pos: List[float],
    init_q: List[float],
    curr_pos: List[float],
    corridor: Union[List[float], str],
    use_xyzw: bool = True,
) -> str:
    """
    Infer current geometric corridor state.

    IMPORTANT:
    This function intentionally does NOT use target_pos, future waypoints,
    stop_ready, or terminal distance. Therefore the same state can be computed
    during online inference.

    Output:
      corridor_centered
      corridor_left_deviation
      corridor_right_deviation
      corridor_down_approach
    """
    if init_direction == "down" or corridor == "down":
        return "corridor_down_approach"

    corridor_min, corridor_max = float(corridor[0]), float(corridor[1])
    lo = corridor_min - CORRIDOR_TOLERANCE
    hi = corridor_max + CORRIDOR_TOLERANCE

    angle_curr, dist_xy, _ = point_angle_in_initial_body(
        p_init=init_pos,
        q_init=init_q,
        p_world=curr_pos,
        use_xyzw=use_xyzw
    )

    if dist_xy < CORRIDOR_DIST_EPS:
        return "corridor_centered"

    if angle_curr < lo:
        return "corridor_left_deviation"

    if angle_curr > hi:
        return "corridor_right_deviation"

    return "corridor_centered"



def infer_current_corridor_state_with_prev_prediction(
    init_direction: str,
    init_pos,
    init_q,
    curr_pos,
    corridor,
    use_xyzw: bool = True,
) -> str:
    """
    Online corridor state with one-step memory.

    Rule:
        If previous model prediction strongly indicates terminal convergence,
        then current corridor state = corridor_terminal_zone.

    Otherwise:
        Use target-free odometry-based corridor state.
    """

    return infer_current_corridor_state_online_no_target(
        init_direction=init_direction,
        init_pos=init_pos,
        init_q=init_q,
        curr_pos=curr_pos,
        corridor=corridor,
        use_xyzw=use_xyzw,
    )


# ============================================================
# Output parsing
# ============================================================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from model output.
    """
    text = text.strip()

    # Some decoded outputs may still contain assistant marker.
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1].strip()

    if "assistant" in text:
        text = text.split("assistant")[-1].strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: extract from first "{" to last "}"
    left = text.find("{")
    right = text.rfind("}")

    if left != -1 and right != -1 and right > left:
        json_str = text[left:right + 1]
        return json.loads(json_str)

    raise ValueError(f"Cannot parse JSON from output: {text}")


def normalize_waypoints_body(
    waypoints_body,
    k_future: int = K_FUTURE,
) -> List[List[float]]:
    """
    Ensure waypoints_body is a list with exactly K waypoints,
    and each waypoint has 3 float values.
    """
    default_wp = [[0.0, 0.0, 0.0] for _ in range(k_future)]

    if waypoints_body is None:
        return default_wp

    if not isinstance(waypoints_body, list):
        return default_wp

    cleaned = []

    for wp in waypoints_body:
        if not isinstance(wp, (list, tuple)) or len(wp) != 3:
            continue

        try:
            cleaned.append([float(wp[0]), float(wp[1]), float(wp[2])])
        except Exception:
            continue

    if len(cleaned) == 0:
        return default_wp

    # pad or trim to exactly K
    if len(cleaned) < k_future:
        last = cleaned[-1]
        while len(cleaned) < k_future:
            cleaned.append(last[:])

    if len(cleaned) > k_future:
        cleaned = cleaned[:k_future]

    return cleaned


def build_fallback_prediction() -> Dict[str, Any]:
    """
    Used when model output cannot be parsed.
    """
    return {
        "target_direction": "front",
        "diagnosis": "forward_progress",
        "action": "forward",
        "stop": "no",
        "waypoints_body": [[0.0, 0.0, 0.0] for _ in range(K_FUTURE)],
    }


# ============================================================
# Inference
# ============================================================

def eval(
    eval_env: AirVLNENV,
    eval_save_dir,
    model,
    processor,
    device,
    system_prompt: str = SYSTEM_PROMPT,
):
    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()

            if env_batchs is None:
                break

            traj_name = [b["seq_name"] for b in env_batchs]

            batch_state = EvalBatchState(
                batch_size=eval_env.batch_size,
                env_batchs=env_batchs,
                env=eval_env,
            )

            print("batch_state---")
            pbar.update(n=eval_env.batch_size)

            # ------------------------------------------------------------
            # Per-batch trajectory state cache.
            # Each sample in the batch has its own initial image / pose / direction.
            # ------------------------------------------------------------
            

            # Previous prediction cache.
            # Used to trigger corridor_terminal_zone at the next step.

            corridor = None
            init_direction = None
            init_pos = None
            init_q = None
            init_front_img  = None
            for t in range(int(args.maxWaypoints) + 1):
                logger.info(
                    "Step: {} \t Completed: {} / {}".format(
                        t,
                        int(eval_env.index_data) - int(eval_env.batch_size),
                        end_iter,
                    )
                )

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break

                # ------------------------------------------------------------
                # Current pose and orientation for each sample
                # ------------------------------------------------------------
                cur_pose = []
                cur_ori = []

                for i in range(eval_env.batch_size):
                    cur_pose.append(eval_env.sim_states[i].pose[:3])
                    cur_ori.append(eval_env.sim_states[i].pose[-4:])

                prompts = []
                images_list = []

                # ------------------------------------------------------------
                # Build batch prompts
                # ------------------------------------------------------------
                for i in range(eval_env.batch_size):
                    data = batch_state.episodes[i][-1]

                    raw_instruction = data["instruction"].strip()
                    instruction = strip_direction_sentence(raw_instruction)

                    front_img = Image.fromarray(data["rgb_record"][0])
                    down_img = Image.fromarray(data["rgb_record"][1])

                    # --------------------------------------------------------
                    # Initialize per-sample state at t=0
                    # --------------------------------------------------------
                    if t == 0:
                        init_front_img = front_img
                        init_pos = cur_pose[i]
                        init_q = cur_ori[i]

                        init_direction = parse_initial_direction_from_instruction(
                            raw_instruction
                        )

                        if init_direction is None:
                            print(
                                "[Warning] Cannot parse initial direction from instruction. "
                                f"Use default front. Raw instruction: {raw_instruction}"
                            )
                            init_direction = "front"

                        corridor = initial_direction_to_corridor(init_direction)


                    current_corridor_state = infer_current_corridor_state_with_prev_prediction(
                        init_direction=init_direction,
                        init_pos=init_pos,
                        init_q=init_q,
                        curr_pos=cur_pose[i],
                        corridor=corridor,
                        use_xyzw=USE_XYZW,
                    )

                    # --------------------------------------------------------
                    # Image order must match training:
                    #   [Current Front, Current Down]
                    # --------------------------------------------------------

                    images_list.append(
                                            [
                                                front_img,
                                                down_img,
                                            ]
                                        )
                    user_prompt = build_user_prompt(
                        instruction=instruction,
                        init_direction=init_direction,
                        corridor=corridor,
                        current_corridor_state=current_corridor_state,
                    )

                    # In messages, images are already provided as image objects.
                    # Remove the literal <image> text to avoid duplicate tokens.
                    user_prompt = user_prompt.replace(
                        "<image>\n<image>\n",
                        "",
                    )

                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": system_prompt,
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "image"},
                                {
                                    "type": "text",
                                    "text": user_prompt,
                                },
                            ],
                        },
                    ]

                    prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    prompts.append(prompt)

                # ------------------------------------------------------------
                # Processor inputs
                # ------------------------------------------------------------
                inputs = processor(
                    text=prompts,
                    images=images_list,
                    return_tensors="pt",
                    padding=True,
                )

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)

                # ------------------------------------------------------------
                # Generate
                # ------------------------------------------------------------
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                    )

                results = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                print("Raw model outputs:")
                print(results)

                # ------------------------------------------------------------
                # Parse model outputs
                # ------------------------------------------------------------
                traj = []
                hover_list = []

                for i, result in enumerate(results):
                    try:
                        data_result = extract_json_from_text(result)
                    except Exception as e:
                        print(f"[JSON parse error] batch={i}, t={t}, error={e}")
                        print("Raw output:", result)
                        data_result = build_fallback_prediction()

                    stop = data_result.get("stop", "no")

                    waypoints_body = normalize_waypoints_body(
                        data_result.get("waypoints_body", None),
                        k_future=K_FUTURE,
                    )

                    traj.append(waypoints_body)


                    if stop == "yes":
                        hover = True
                    else:
                        hover = False
                    hover_list.append(hover)


                # ------------------------------------------------------------
                # Convert predicted body-frame waypoints to world-frame waypoints
                # ------------------------------------------------------------
                global_taj = []
                for i in range(eval_env.batch_size):
                    local_traj = traj[i]
                    if all(v == 0 for row in local_traj for v in row):
                        local_traj.append(cur_pose[i])
                        global_taj.append(local_traj)
                    # print("local_traj",local_traj)
                    else:
                        P_t = np.array(cur_pose[i])
                        q_t = np.array(cur_ori[i])   # (x, y, z, w)
                        R_t = R.from_quat(q_t).as_matrix()  # body -> world
            
                        pred_world = []
                        for dp_body in local_traj:  # dp_body: 单个相对位移
                            dp_world = R_t @ dp_body        # body -> world
                            p_future = P_t + dp_world       # 加上当前全局位置
                            pred_world.append(p_future)
                        pred_world.append(cur_pose[i])
                        global_taj.append(pred_world)
                    # print("pred_world:", pred_world)
                next_waypoints = global_taj

                # ------------------------------------------------------------
                # Execute actions
                # ------------------------------------------------------------
                eval_env.makeActions_qwenvl(next_waypoints)

                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)
                batch_state.update_metric_land(hover_list)

        try:
            pbar.close()
        except Exception:
            pass


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path
    log_path = args.log_path
    port = args.simulator_tool_port
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)


    folder_name = os.path.basename(os.path.normpath(args.eval_save_path))
    log_file = open(
        log_path+"/"+folder_name+".txt",
        "a",
        buffering=1,
    )
    sys.stdout = log_file

    merged_model_path = model_args.model_path

    processor = AutoProcessor.from_pretrained(
        merged_model_path,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        merged_model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )

    model.eval()
    device = model.device

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(
        dataset_path=dataset_path,
        save_path=eval_save_path,
        eval_json_path=eval_json_path,
        port=port,
    )

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    eval(
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        model=model,
        processor=processor,
        device=device,
        system_prompt=SYSTEM_PROMPT,
    )

    eval_env.delete_VectorEnvUtil()
