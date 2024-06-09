import os
import xml.etree.ElementTree as ET
import json

# XML 파일을 COCO 형식으로 변환하는 함수
def convert_xml_to_coco(xml_files, output_json):
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "hand"},
            {"id": 2, "name": "black_key"},
            {"id": 3, "name": "white_key"}
        ]
    }
    annotation_id = 1

    for idx, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 이미지 정보
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        image_info = {
            "id": idx + 1,
            "file_name": filename,
            "width": width,
            "height": height
        }
        dataset["images"].append(image_info)

        # 어노테이션 정보
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = next((item["id"] for item in dataset["categories"] if item["name"] == class_name), None)

            if class_id is None:
                continue

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            annotation = {
                "id": annotation_id,
                "image_id": idx + 1,
                "category_id": class_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            }
            dataset["annotations"].append(annotation)
            annotation_id += 1

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(dataset, f)

# XML 파일이 있는 디렉토리 경로
xml_dir = './data/Keyboard/labeledxml'
xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]

# 변환 결과를 저장할 JSON 파일 경로
output_json = './data/Keyboard/jsonResult/dataset.json'

# XML 파일을 COCO 형식으로 변환
convert_xml_to_coco(xml_files, output_json)

print("XML 파일을 COCO 형식으로 변환 완료!")
