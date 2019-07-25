import re
# keypoints (FloatTensor[N, K, 3]): the locations of the predicted keypoints, in [x, y, v] format.
# During training, the model expects both the input tensors, as well as a targets (list of dictionary),
#    containing:
#        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
#          between 0 and H and 0 and W
#        - labels (Int64Tensor[N]): the class label for each ground-truth box
#        - keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.

annotation_path = 'Datasets/Synthetic_Wolf_1__2019_07_18/Annotation/BonePositions.txt'
file = open(annotation_path, "r")

target = []
labels = {}
keypoints = {}
boxes = {}
count = 0

for i, line in enumerate(file):
    # Extracting labels, Labels of keypoints are stored in the first index: index 0
    if i == 0:
        for j, keypoint_label in enumerate(line.split(' ')):
            if j is not 0:
                labels[j] = keypoint_label
    # Exracting keypoints, keypoints are stored after the first line of the file
    individual_cordinates = line.split(' ')[0]
    individual_cordinates = re.split(r'\t+', individual_cordinates)
    print("Processing Metadata: " + str(int(i*100/34301)) + "%", end='\r')
    for j, cordinate in enumerate(individual_cordinates):
        count += 1        
        # Skip even iterations as including it duplicates (x,y) cordinates entry
        if j%2 == 0:
            continue
        # Skip the value in the first index of the list as this is the frame number    
        if j is not 0:
            keypoints[int(count / 2)] = [i, [float(cordinate), float(individual_cordinates[j + 1]), 1], 3] 
            if j is 1:
                # Obtain the boxes of each samples, theese are: xmin, xmax, ymin, ymax
                xmin = float(cordinate)
                xmax = float(cordinate)
                ymin = float(individual_cordinates[j + 1])
                ymax = float(individual_cordinates[j + 1])
                boxes[i] = [i, [xmin, ymin, xmax, ymax]]
            else:
                xmin = xmin if (xmin < float(cordinate)) else float(cordinate)
                xmax = xmax if (xmax > float(cordinate)) else float(cordinate)
                ymin = ymin if (ymin < float(individual_cordinates[j + 1])) else float(individual_cordinates[j + 1])
                ymax = ymax if (ymax > float(individual_cordinates[j + 1])) else float(individual_cordinates[j + 1])
                boxes[i] = [i, [xmin, ymin, xmax, ymax]]

target.append(labels)
target.append(keypoints)
target.append(boxes)
