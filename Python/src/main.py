#Starting code for a project to detect pigweed in a field
import cv2
import os

def read_annotations(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        bounding_boxes = []
        for line in lines:
            values = line.split()
            class_id, center_x, center_y, width, height = map(float, values)
            bounding_boxes.append([class_id, center_x, center_y, width, height])
        return bounding_boxes

def draw_annotations(img, bounding_boxes, save_path):
    image = cv2.imread(img)
    if image is None:
        print("Could not read image")
        exit()
    h, w, _ = image.shape
    for bounding_box in bounding_boxes:
        class_id, center_x, center_y, width, height = bounding_box
        x1 = int((center_x - width / 2) * w)
        y1 = int((center_y - height / 2) * h)
        x2 = int((center_x + width / 2) * w)
        y2 = int((center_y + height / 2) * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
    cv2.imwrite(save_path, image)

    
image_folder = 'Training Data\PigweedDataSet\images'
annotation_folder = 'Training Data\PigweedDataSet\\annotations'
output_folder = "Training Data\PigweedDataSet\\annotated_images/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for i in range(1, os.listdir(image_folder).__len__() + 1):
    image_name = f"pigweed_{i:03d}.png"
    txt_name = f"pigweed_{i:03d}.txt"
    
    image_path = os.path.join(image_folder, image_name)
    annotation_path = os.path.join(annotation_folder, txt_name)
    save_path = os.path.join(output_folder, image_name)
    
    # Check if the .txt file exists before proceeding
    if not os.path.exists(annotation_path):
        continue
    
    bounding_boxes = read_annotations(annotation_path)
    draw_annotations(image_path, bounding_boxes, save_path)
    

