"""
Process HaMeR: Single-camera hand reconstruction using HaMeR.

Reconstructs hand mesh from a single camera view for comparison
against the multi-view MANO fitting pipeline (process_hand_fitting.py).
"""

import os
import sys
import argparse
import pickle
from uuid import uuid4

# Add parent directory for utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Add hamer directory for HaMeR and ViTPose imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hamer'))

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
_np_patches = {
    'bool': bool, 'int': int, 'float': float, 'complex': complex,
    'object': object, 'unicode': str, 'str': str,
}
for _name, _replacement in _np_patches.items():
    if not hasattr(np, _name):
        setattr(np, _name, _replacement)

from pathlib import Path
import torch
import cv2
import rerun as rr
import rerun.blueprint as rrb

from hand_object_tracking.utils.camera_util import loadAllLatestIntrinsic, get_total_num_gpus, get_undistort_maps, undistort_image
from hand_object_tracking.utils.h264Util import StreamRecordReader

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def _setup_detector(body_detector: str):
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer as _hamer_pkg
        cfg_path = Path(_hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/"
            "f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config(
            'new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)
    raise ValueError(f"Unknown body detector: {body_detector}")


def process_hamer(
    camera_id: str,
    recording: str,
    markerless_data_dir: str,
    num_frames: int = 1,
    frame_start: int = 0,
    out_folder: str = 'out_hamer',
    checkpoint: str = DEFAULT_CHECKPOINT,
    batch_size: int = 1,
    save_mesh: bool = False,
    body_detector: str = 'vitdet',
    rescale_factor: float = 2.0,
    recording_id: str = None,
    use_rerun: bool = True,
    undistort: bool = True,
    side_view: bool = False,
    viz_scale: float = 1000.0,
):
    """
    Run HaMeR hand reconstruction on frames from a single camera.

    Args:
        camera_id: Serial number of the camera to use (e.g. '233500779')
        recording: Recording folder name (e.g. '2026-01-30-02-09-22_rerunViz')
        markerless_data_dir: Path to the markerless data root directory
        num_frames: Number of frames to process (None = all available)
        frame_start: First frame index to process
        out_folder: Output directory for rendered overlay images
        checkpoint: Path to HaMeR model checkpoint
        batch_size: Batch size for HaMeR inference
        save_mesh: If True, save hand meshes as OBJ files
        body_detector: Body detector to use ('vitdet' or 'regnety')
        rescale_factor: Padding multiplier applied to the hand bounding box
        recording_id: Rerun recording ID (None = random UUID)
        use_rerun: If True, log results to Rerun for 3D visualisation
        undistort: If True, undistort frames using camera calibration
        side_view: If True, render a side view for each detected hand crop
        viz_scale: Unit scale for Rerun (default 1000.0 converts metres to mm)
    """
    print(f"[HaMeR] Starting reconstruction for camera {camera_id}...")

    os.makedirs(out_folder, exist_ok=True)

    # ---- Load camera calibration ----
    intrinsic_folder = os.path.join(markerless_data_dir, "intrinsics") + "/"
    intrinsics = loadAllLatestIntrinsic(intrinsic_folder)
    if camera_id not in intrinsics:
        raise ValueError(f"Camera {camera_id} not found in intrinsics. Available: {sorted(intrinsics.keys())}")

    extrinsic_file = os.path.join(recording, "systemCalibration.pkl")
    with open(extrinsic_file, 'rb') as f:
        extrinsics_dict = pickle.load(f)
    if camera_id not in extrinsics_dict:
        raise ValueError(f"Camera {camera_id} not found in extrinsics. Available: {sorted(extrinsics_dict.keys())}")

    compressed_meta_file = os.path.join(
        recording, "compressed", "compressMetadata.pkl"
    )
    with open(compressed_meta_file, 'rb') as f:
        compressed_data = pickle.load(f)

    img_w = compressed_data['imgW']
    img_h = compressed_data['imgH']
    total_frames = len(compressed_data['seekTable'][0])
    if num_frames is None:
        num_frames = total_frames - frame_start
    num_frames = min(num_frames, total_frames - frame_start)
    print(f"[HaMeR] Processing {num_frames} frames starting from {frame_start} (total available: {total_frames})")

    # ---- Camera intrinsics / extrinsics ----
    K = intrinsics[camera_id]['K']
    D = intrinsics[camera_id]['D']
    new_K, map1, map2, _roi = get_undistort_maps(K, D, img_w, img_h, alpha=0)
    K_to_use = new_K if undistort else K

    # Camera-to-world transform (extrinsics are camFrame_labFrame = world-to-camera)
    T_world_camera = np.linalg.inv(extrinsics_dict[camera_id]['camFrame_labFrame'])
    R_cw = T_world_camera[:3, :3]  # camera-to-world rotation
    t_cw = T_world_camera[:3, 3]   # camera-to-world translation (metres)

    # ---- Setup video reader ----
    compressed_folder = os.path.join(
        recording, "compressed"
    ) + "/"
    gpu_no = get_total_num_gpus()
    gpu_list = [-1, -1] if gpu_no == 0 else ([0, 0] if gpu_no == 1 else [0, 1])
    record_reader = StreamRecordReader(compressed_folder, gpu_list)

    if camera_id not in record_reader.camIdList:
        raise ValueError(f"Camera {camera_id} not in compressed data. Available: {record_reader.camIdList}")
    cam_img_index = record_reader.camIdList.index(camera_id)

    # ---- Setup HaMeR model ----
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[HaMeR] Using device: {device}")
    model, model_cfg = load_hamer(checkpoint)
    model = model.to(device)
    model.eval()

    detector = _setup_detector(body_detector)
    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # HaMeR uses a synthetic focal length during training (EXTRA.FOCAL_LENGTH = 5000 at
    # MODEL.IMAGE_SIZE = 224). cam_crop_to_full scales it proportionally to the full image.
    # All predicted cam_t values are consistent with this virtual camera, NOT with the real K.
    hamer_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_w, img_h)
    K_hamer = np.array([
        [hamer_focal_length, 0, img_w / 2.0],
        [0, hamer_focal_length, img_h / 2.0],
        [0, 0, 1.0]], dtype=np.float64)

    # ---- Setup Rerun ----
    stream = None
    if use_rerun:
        if recording_id is None:
            recording_id = str(uuid4())
        stream = rr.RecordingStream("HaMeR Reconstruction", recording_id=recording_id)
        stream.connect_grpc()

        # stream.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        print(f"[HaMeR] Rerun recording ID: {recording_id}")

        stream.set_time("frame_count", sequence=0)

        t_rr = t_cw * viz_scale
        # Log HaMeR's virtual camera so that (verts + cam_t) project correctly onto
        # the image in Rerun's 2D view.  The real K is intentionally NOT used here.
        stream.log(f"cameras/camera_{camera_id}", rr.Pinhole(
            image_from_camera=K_hamer,
            resolution=[img_w, img_h],
        ))
        stream.log(f"cameras/camera_{camera_id}", rr.Transform3D(translation=t_rr, mat3x3=R_cw))
        stream.log(f"world/camera_point_{camera_id}",
                   rr.Points3D(np.array([t_rr]), colors=[255, 0, 0], radii=0.005 * viz_scale))

    # ---- Process frames ----
    for frame_idx in range(frame_start, frame_start + num_frames):
        seq = frame_idx - frame_start
        print(f"[HaMeR] Frame {frame_idx} ({seq + 1}/{num_frames})...")

        images = record_reader.readFrame(frame_idx)
        img_cv2 = images[cam_img_index]
        if img_cv2 is None:
            print(f"[HaMeR]   -> no image data, skipping")
            continue

        # Undistort image without cropping to preserve K alignment
        if undistort:
            img_cv2 = undistort_image(img_cv2, map1, map2)

        img_rgb = img_cv2[:, :, ::-1].copy()  # BGR -> RGB for ViTPose

        # ---- Person detection ----
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        def _log_raw_image():
            if stream is not None:
                stream.set_time("frame_count", sequence=seq)
                _, jpeg_bytes = cv2.imencode('.jpg', img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 80])
                stream.log(f"cameras/camera_{camera_id}/image", rr.EncodedImage(
                    contents=jpeg_bytes.tobytes(), media_type="image/jpeg", opacity=0.5
                ))

        if len(pred_bboxes) == 0:
            print(f"[HaMeR]   -> no persons detected")
            _log_raw_image()
            continue

        # ---- Hand keypoint detection ----
        vitposes_out = cpm.predict_pose(
            img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        bboxes = []
        is_right_list = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            for keyp, hand_right in [(left_hand_keyp, 0), (right_hand_keyp, 1)]:
                valid = keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                            keyp[valid, 0].max(), keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right_list.append(hand_right)

        # print(f'Keypoint shape: left {left_hand_keyp.shape}, right {right_hand_keyp.shape}')
        # print(f'Example of left hand keypoints (x, y, conf): {left_hand_keyp[:5]}')

        if len(bboxes) == 0:
            print(f"[HaMeR]   -> no hands detected")
            _log_raw_image()
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right_list)

        # ---- HaMeR inference ----
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        all_verts = []
        all_cam_t = []
        all_right_flags = []
        all_keypoints_3d = []
        scaled_focal_length = None

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_keypoints_3d = out['pred_keypoints_3d'].detach()
            pred_keypoints_3d = pred_keypoints_3d[:,None,:,:]

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            for n in range(batch['img'].shape[0]):
                person_id = int(batch['personid'][n])
                input_patch = (
                    batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255)
                    + (DEFAULT_MEAN[:, None, None] / 255)
                ).permute(1, 2, 0).numpy()

                # Per-crop rendered overlay
                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )

                if side_view:
                    white_img = (
                        (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255)
                        / (DEFAULT_STD[:, None, None] / 255)
                    )
                    side_img = renderer(
                        out['pred_vertices'][n].detach().cpu().numpy(),
                        out['pred_cam_t'][n].detach().cpu().numpy(),
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                    )
                    crop_result = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    crop_result = np.concatenate([input_patch, regression_img], axis=1)

                cv2.imwrite(
                    os.path.join(out_folder, f'frame_{frame_idx:06d}_{person_id}.png'),
                    255 * crop_result[:, :, ::-1]
                )

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_flag = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_flag - 1) * verts[:, 0]

                cam_t = pred_cam_t_full[n]
                # cam_t[2] = cam_t[2] / 1000.0
                print(f'CAM_T : {cam_t}')
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right_flags.append(is_right_flag)

                keyp_3d = pred_keypoints_3d[n, 0].cpu().numpy()
                keyp_3d[:, 0] = (2 * is_right_flag - 1) * keyp_3d[:, 0]
                all_keypoints_3d.append(keyp_3d)

                if save_mesh:
                    tmesh = renderer.vertices_to_trimesh(
                        verts, cam_t.copy(), LIGHT_BLUE, is_right=is_right_flag
                    )
                    tmesh.export(os.path.join(out_folder, f'frame_{frame_idx:06d}_{person_id}.obj'))

        # Full-frame overlay image
        overlay = None
        if len(all_verts) > 0 and scaled_focal_length is not None:
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[n],
                is_right=all_right_flags,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            input_float = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            overlay = input_float * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            cv2.imwrite(
                os.path.join(out_folder, f'frame_{frame_idx:06d}_all.jpg'),
                255 * overlay[:, :, ::-1]
            )

        # ---- Log to Rerun ----
        if stream is not None:
            stream.set_time("frame_count", sequence=seq)

            # Camera image: use overlay when available, else raw frame
            # if overlay is not None:
            #     overlay_bgr = (255 * overlay[:, :, ::-1]).clip(0, 255).astype(np.uint8)
            #     _, jpeg_bytes = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            # else:
            _, jpeg_bytes = cv2.imencode('.jpg', img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 80])
            stream.log(f"cameras/camera_{camera_id}/image", rr.EncodedImage(
                contents=jpeg_bytes.tobytes(), media_type="image/jpeg", opacity=0.5
            ))

            # 3D hand meshes and keypoints in world space
            for hand_idx, (verts, cam_t, is_right_flag, keyp_3d) in enumerate(
                zip(all_verts, all_cam_t, all_right_flags, all_keypoints_3d)
            ):
                # verts and cam_t are in HaMeR's virtual camera space — the coordinate
                # system defined by the synthetic focal length, not real-world metres.
                # Do NOT apply viz_scale: these are already at the right numerical scale
                # for the virtual Pinhole camera logged above.
                verts_cam = verts + cam_t[None, :]
                verts_world = (R_cw @ verts_cam.T).T + t_cw

                hand_label = "right" if is_right_flag > 0.5 else "left"
                stream.log(
                    f"world/hamer_{hand_label}_hand_{hand_idx}",
                    rr.Mesh3D(
                        vertex_positions=verts_world,
                        triangle_indices=model.mano.faces,
                        vertex_colors=[100, 180, 220],
                    )
                )

                keyp_cam = keyp_3d + cam_t[None, :]
                keyp_world = (R_cw @ keyp_cam.T).T + t_cw
                stream.log(
                    f"world/hamer_{hand_label}_keypoints_{hand_idx}",
                    rr.Points3D(keyp_world, colors=[255, 200, 0], radii=0.002)
                )

    # ---- Rerun blueprint ----
    if stream is not None:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="3D Scene", contents=["/**"]),
                rrb.Spatial2DView(
                    origin=f"cameras/camera_{camera_id}",
                    name=f"Camera {camera_id}",
                    contents=["+ $origin/**", "+ world/**"],
                ),
            )
        )
        stream.send_blueprint(blueprint)

    print(f"[HaMeR] Done. Results saved to '{out_folder}'")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, '..', 'data', 'markerless_data')

    parser = argparse.ArgumentParser(
        description='HaMeR single-camera hand reconstruction for pipeline comparison'
    )

    # Required
    parser.add_argument('--camera-id', required=True,
                        help='Camera serial number (e.g. 233500779)')
    parser.add_argument('--recording-name', required=True,
                        help='Recording folder name (e.g. 2026-01-30-02-09-22_rerunViz)')

    # Frame selection
    parser.add_argument('--num-frames', type=int, default=1,
                        help='Number of frames to process (default: all available)')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Index of the first frame to process (default: 0)')

    # Output
    parser.add_argument('--out-folder', default='out_hamer',
                        help='Output directory for rendered overlay images (default: out_hamer)')
    parser.add_argument('--save-mesh', action='store_true', default=False,
                        help='Save reconstructed hand meshes as OBJ files')

    # Model
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='Path to HaMeR model checkpoint')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for HaMeR inference (default: 1)')
    parser.add_argument('--body-detector', default='vitdet', choices=['vitdet', 'regnety'],
                        help='Body detector backend (default: vitdet)')
    parser.add_argument('--rescale-factor', type=float, default=2.0,
                        help='Padding multiplier applied to the hand bounding box (default: 2.0)')

    # Image preprocessing
    parser.add_argument('--no-undistort', action='store_true', default=False,
                        help='Disable lens undistortion (undistortion is on by default)')
    parser.add_argument('--side-view', action='store_true', default=False,
                        help='Render a side-view crop for each detected hand')

    # Rerun
    parser.add_argument('--recording-id',
                        help='Rerun recording ID to attach to (default: new random UUID)')
    parser.add_argument('--no-rerun', action='store_true', default=False,
                        help='Disable Rerun logging; only save images to disk')
    parser.add_argument('--viz-scale', type=float, default=1000.0,
                        help='Scale factor for Rerun units (default: 1000.0, metres -> mm)')

    args = parser.parse_args()

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

    # ==== Camera Visualization Process ====
    import glob
    RECORDING_NAME = args.recording_name
    MARKERLESS_DATA_DIR = os.path.join(DATA_DIR, "markerless_data")
    MARKERLESS_DATA_FOLDER = glob.glob(os.path.join(MARKERLESS_DATA_DIR, "recordings", f"*_{RECORDING_NAME}"))[0]


    process_hamer(
        camera_id=args.camera_id,
        recording=MARKERLESS_DATA_FOLDER,
        markerless_data_dir=MARKERLESS_DATA_DIR,
        num_frames=args.num_frames,
        frame_start=args.frame_start,
        out_folder=args.out_folder,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        save_mesh=args.save_mesh,
        body_detector=args.body_detector,
        rescale_factor=args.rescale_factor,
        recording_id=args.recording_id,
        use_rerun=not args.no_rerun,
        undistort=not args.no_undistort,
        side_view=args.side_view,
        viz_scale=args.viz_scale,
    )


if __name__ == '__main__':
    main()
