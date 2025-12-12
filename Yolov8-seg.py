from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error, camera not found')
    exit()


while cap.isOpened():
    success,frame = cap.read()

    if not success:
        break

    results = model(frame,stream=True,verbose=False)

    for result in results:
        if result.masks is not None and len(result.masks.data) > 0:

            # Combine all detected masks into a single mask (the key step!)

            # Sum the masks along the channel dimension to combine all instances (bike,people,houses)
            combined_mask = np.sum(result.masks.data.cpu().numpy(),axis=0)

            # Resizing the image
            H,W = frame.shape[:2]

            mask_resized = cv2.resize(combined_mask,(W,H),interpolation=cv2.INTER_NEAREST)

            # Threshold the mask to get binary pixels (0 or 1)

            # Convert the probability (0.0 to 1.0) into a hard 0 or 255 mask
            mask_float = (mask_resized > 0.5).astype(float)

            # Implementing a simple background blur

            # Create a blurred background
            blurred_frame = cv2.GaussianBlur(frame,(51,51),0)

            # Expand the mask to 3 channels
            mask_3ch_float = mask_resized[:,:,None]

            # Calculate foreground (the person)
            foreground  = frame.astype(float) * mask_3ch_float

            # Calculate background
            background = blurred_frame.astype(float) * (1-mask_3ch_float)

            # Combine both the foreground and the background
            final_output = cv2.add(foreground,background).astype(np.uint8)

            cv2.imshow('Real-Time segmentation demo', final_output)

        else:
            cv2.imshow('Real-Time segmentation demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



