import openvr
import time
import json
import cv2
import pylab as plt
import numpy as np
from scipy.spatial.transform import Rotation

from reconstruct import ColmapFolder
from vendor import triad_openvr 
from vendor import read_write_model 
from vendor import database

# Settings from config.json
config = json.load(open("config.json","r"))
blur_threshold=config["settings"]["blur_rejection_threshold"]
interval=config["settings"]["interval_in_seconds"]
device_name=config["settings"]["use_device"]
camera_index=config["settings"]["camera_index"]
colors = config["colors"]

# Output
output_directory = "./OUTPUT"
db = database.COLMAPDatabase.connect(ColmapFolder(output_directory).database_path)
db.create_tables()

# Draws axes on the plot
def draw_axes(ax, transform_mat, **kwargs):
  axis = np.zeros((3, 4))
  axis[0] = transform_mat @ np.array([1, 0, 0, 0]).T
  axis[1] = transform_mat @ np.array([0, 1, 0, 0]).T
  axis[2] = transform_mat @ np.array([0, 0, 1, 0]).T
  pos     = transform_mat @ np.array([0, 0, 0, 1]).T
  ax.quiver(pos[0], pos[2], pos[1], axis[:, 0], axis[:, 2], axis[:, 1], normalize=True, length=0.15, colors=['r', 'g', 'b', 'r', 'r', 'g', 'g', 'b', 'b'])

# Coverts euler pose to multidimensional array
def pose_matrix_to_numpy(pose_matrix):
    try:
        pose_arr = np.zeros((4, 4))
        pose_arr[0] = pose_matrix[0]
        pose_arr[1] = pose_matrix[1]
        pose_arr[2] = pose_matrix[2]
        pose_arr[3] = [0, 0, 0, 1]
        return pose_arr
    except:
        return []


###################################################################################
print("\n\nLooking for MFPucks...")
print("==============================================")
v = triad_openvr.triad_openvr("config.json")
v.print_discovered_objects()
vc = cv2.VideoCapture(camera_index)

print("\n\nInitializing Camera...")
print("==============================================")


# Init Plot (graph image positions)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Meters (X)')
ax.set_ylabel('Meters (Y)')
ax.set_zlabel('Meters (Z)')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0, 1)


# Init webcam preview window and grab first frame
cv2.namedWindow("preview")
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Cam setup
# FIXME: A lot of this conversion/transform/git mat code is mysterious to me... it comes from openvr_camera
convert_coordinate_system = np.identity(4)
convert_coordinate_system[:3, :3] = Rotation.from_euler('XYZ',(180, 0, 0), degrees=True).as_matrix()
camera_model = read_write_model.CAMERA_MODEL_NAMES['SIMPLE_PINHOLE']
cameras = {}
camera_to_puck_transforms = {}
num_cameras=1 # Headsets have 2 (openvr.VRSystem().getInt32TrackedDeviceProperty(device, openvr.Prop_NumCameras_Int32))
camera_to_puck_mat = (openvr.HmdMatrix34_t*num_cameras) ()
height, width = frame.shape[:2]
cv2.resizeWindow("preview", height,width)
init_params = np.array((420.000000, (width/num_cameras)/2, height/2, 0.000000))
for i in range(num_cameras):
    cam_id = db.add_camera(camera_model.model_id, width/2, height, init_params)
    camera_to_puck_transforms[cam_id] = pose_matrix_to_numpy(camera_to_puck_mat[i])
    cameras[cam_id] = read_write_model.Camera(id=cam_id, model=camera_model.model_name, width=width/num_cameras, height=height, params=init_params)
camera_to_puck_mat = (openvr.HmdMatrix34_t*num_cameras) ()

image_count=0
elapsed = interval
while rval:
    rval, frame = vc.read()
    if(elapsed >= interval):
        # Blur detection also detects black frames, this is useful because you can "pause" shooting by covering the lens.
        # Mostly I want it because I'd like to feed this video frame-grabs and need to ensure those aren't blurry.
        
        # https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        focus_measure =  cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = (focus_measure < blur_threshold)
        text = "Blurry!" if is_blurry else "OK!" 
        color = colors["red"] if is_blurry else colors["green"]
        cv2.putText(
            frame, 
            "{}: {:.2f}".format(text, focus_measure), 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color["B"], color["G"], color["R"]),3)
        cv2.imshow("preview", frame)

        # Reset time, record position
        start = time.time()
        if(device_name in v.devices and not is_blurry):
            # FIXME: A lot of this matrix transform code is mysterious to me... it comes from openvr_camera
            # I suspect it is not correct for the puck (vs the HMD where it came from) Also possibly the euler pose
            # is not what it's expecting in the first place?
            puck_pose_arr = v.devices[device_name].get_pose_euler()
            world_to_puck = pose_matrix_to_numpy(puck_pose_arr)
            world_to_cams = {id_:world_to_puck @ head_to_cam @ convert_coordinate_system for (id_,head_to_cam) in camera_to_puck_transforms.items()}
            for j, (cam_id, world_to_cam) in enumerate(world_to_cams.items()):
                draw_axes(ax, transform_mat=world_to_puck)
                # FIXME: Should add to db (see db.add_image in openvr_camera.py)
                # Record frame to image folder
                name = f"{image_count:03d}_cam{j}.jpg"
                print(name)
                path = output_directory + "/images/" + name
                cv2.imwrite(path, original_frame)
                image_count=image_count+1
            fig.show()
            fig.canvas.draw()
            fig.canvas.flush_events()
    elapsed = time.time() - start

# Cleanup
db.commit()
db.close()
plt.show()
vc.release()
openvr.shutdown()
cv2.destroyWindow("preview")