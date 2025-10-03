import os
import argparse
from PIL import Image
import numpy as np


def load_image(fname):
    img = Image.open(fname)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def save_image(npdata, fname) :
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"),"L")
    img.save(fname)

def process_HL_action_img(filename, HL_actions=3, resolution=64):
    """
    Expects a .png file depicting "hlwm_next_context_recon" from Tensorboard where:
      - Height = seq_len * resolution
      - Width  = (HL_actions + 3) * resolution
    """
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="uint8")  # (H, W, 3)

    seq_len = data.shape[0] // resolution
    num_components = HL_actions + 3  # + input + next step prediction + sampled high level action
    assert data.shape[1] == num_components * resolution, "Width does not match expected size."

    data = data.reshape(
        seq_len, resolution,  # time steps
        num_components, resolution,  # components
        3  # channels
    )
    data = data.transpose(2, 0, 1, 3, 4)  # (components, time, height, width, channels)
    return data

def np_to_gif(imgs, out_dir, filename, start_t=0, end_t=None, duration=50, loop=0):
    """
    imgs: numpy array of shape (T, H, W, C) with dtype uint8
    start_t/end_t: slice indices along T (end_t can be None)
    """
    imgs = [Image.fromarray(img) for img in imgs[start_t:end_t]]
    file_out_dir = os.path.join(out_dir, f'{filename}.gif')
    # duration is the number of milliseconds between frames
    imgs[0].save(file_out_dir, save_all=True, append_images=imgs[1:], duration=duration, loop=loop)

def main():
    parser = argparse.ArgumentParser(
        description="Generate GIFs from HL action prediction sprite sheet."
    )

    parser.add_argument("--pred-file", type=str, required=True,
                        help="Path to the input PNG sprite sheet.")
    parser.add_argument("--out-path", type=str, required=True,
                        help="Directory to write GIFs to.")
    parser.add_argument("--hl-act-dim", type=int, default=3,
                        help="Number of high-level action dimensions.")
    parser.add_argument("--dur", type=int, default=200,
                        help="Frame duration in ms for GIFs.")
    parser.add_argument("--start-t", type=int, default=1,
                        help="Start time index for slicing frames.")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Image resolution.")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.pred_file):
        raise FileNotFoundError(f"pred-file not found: {args.pred_file}")

    # Load and process
    imgs = process_HL_action_img(args.pred_file, HL_actions=args.hl_act_dim, resolution=args.resolution)

    # Ensure output dir exists
    os.makedirs(args.out_path, exist_ok=True)

    # Create GIFs
    np_to_gif(imgs[0], args.out_path, "input", args.start_t, -1, args.dur)
    # for ll pred we shift time index by 1
    np_to_gif(imgs[1], args.out_path, "ll_pred", args.start_t + 1, None, args.dur)

    # visualize all HL actions
    for hl_i in range(args.hl_act_dim):
        np_to_gif(imgs[2 + hl_i], args.out_path, f"HL_act_{hl_i}", args.start_t, -1, args.dur)

    # visualize the sampled HL action
    np_to_gif(imgs[-1], args.out_path, "HL_pred", args.start_t, -1, args.dur)


if __name__ == "__main__":
    main()