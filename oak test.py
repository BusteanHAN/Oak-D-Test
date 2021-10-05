import cv2
import depthai

pipeline = depthai.Pipeline()

# Create nodes, configure them and link them together

# Upload the pipeline to the device
with depthai.Device(pipeline) as device:
  # Start the pipeline that is now on the device
  device.startPipeline()

cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

with depthai.Device(pipeline) as device:
    # device = depthai.Device(pipeline, usb2mode=True)
    q_rgb = device.getOutputQueue("rgb")
    frame = None
    detections = []

while True:
    in_rgb = q_rgb.tryGet()
    if in_rgb is not None:
        frame = in_rgb.getCvFrame()
    if frame is not None:
        cv2.imshow("preview", frame)
