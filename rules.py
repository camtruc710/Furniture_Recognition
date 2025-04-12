from efficient_apriori import apriori
from ultralytics import YOLO
import os
import cv2

model = YOLO('E:/Yolov8/Nien_luan1/runs/detect/weights/last.pt')
image_dir_path = "E:/imgs_livingroom/"
image_names_path = []

for f in os.listdir(image_dir_path):
    file_path = os.path.join(image_dir_path, f)
    if os.path.isfile(file_path):
        image_names_path.append(file_path)
# print(image_names_path)
def label_names(image_path):
    frame = cv2.imread(image_path)
    results = model.predict(source=frame, imgsz=640, conf=0.6)
    lnames = set()
    for result in results:
        indices = result.boxes.cls
        for e in indices:
            lnames.add(result.names[int(e)])
    return tuple(lnames)
def rules_furniture(transactions, image_names_path) :
    for img_name in image_names_path:
        transactions.append(label_names(img_name))
    _, rules = apriori(transactions, min_support=0.4, min_confidence=0.6)
    return rules

# def print_rules():
#     transactions = list()
#     rules= rules_furniture(transactions, image_names_path)
#     print("Tập luật:")
#     for rule in rules:
#         print(rule)

def apply_rules(input_set):
    transactions = list()
    rules = rules_furniture(transactions, image_names_path)
    conclusions = set()
    rule_indices = []
    detailed_conclusions = []

    input_str = ", ".join(input_set)
    rules_str = f"Các đồ nội thất: {input_str}\n\nTập luật:\n"

    for idx, rule in enumerate(rules, 1):
        antecedent, consequent = rule.lhs, rule.rhs
        antecedent_set = set(antecedent)

        # Lưu luật nếu thỏa điều kiện
        if antecedent_set.issubset(input_set):
            rule_indices.append(str(idx))
            conclusions.add(consequent)

            # Ghi rõ mối quan hệ cho từng luật
            if len(antecedent) == 1 and len(consequent) == 1:
                detailed_conclusions.append(
                    f"- Khi có '{antecedent[0]}', thường sẽ có '{consequent[0]}'"
                )
            else:
                ant = " + ".join(antecedent)
                cons = ", ".join(consequent)
                detailed_conclusions.append(
                    f"- Khi có '{ant}', thường sẽ có '{cons}'"
                )

        # Ghi toàn bộ luật vào phần đầu
        antecedent_str = ', '.join(antecedent)
        consequent_str = ', '.join(consequent)
        support_str = str(round(rule.support, 3))
        confidence_str = str(round(rule.confidence, 3))
        rule_str = f"({antecedent_str}) -> ({consequent_str}) với support: {support_str}, confidence: {confidence_str}\n"
        rules_str += rule_str

    if not conclusions:
        return rules_str + "\nKhông có luật nào được sinh ra."

    conclusion_summary = f"\nDựa trên các luật {', '.join(rule_indices)}, kết luận:\n" + "\n".join(detailed_conclusions)
    return rules_str + conclusion_summary

# input_set = {'chandelier'}
# print(apply_rules(input_set))
