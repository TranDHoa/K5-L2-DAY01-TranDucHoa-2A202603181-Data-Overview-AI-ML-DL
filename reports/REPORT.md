# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này mô tả cả bức ảnh. Vì record đang thực hiện bài toán Image Classification, phân tích toàn bộ nội dung ảnh và đưa ra danh sách các lớp có khả năng phù hợp 

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách tên lớp được xác định bởi taxonomy đi kèm checkpoint đã huấn luyện, cụ thể là ImageNet-1K. Model không tự tạo ra tên lớp mới mà dự đoán trong tập 1.000 lớp đã được định nghĩa trước.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Cần lưu cả ba trường để đảm bảo khả năng nhận diện và truy xuất thông tin chính xác. class_id là mã định danh lớp, class_name giúp con người hiểu nội dung dự đoán, còn taxonomy_name xác định hệ thống phân loại đang được sử dụng. Điều này giúp tránh nhầm lẫn khi các model sử dụng những taxonomy khác nhau hoặc có cách đánh số class khác nhau.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Cần lưu cả ba trường để đảm bảo khả năng nhận diện và truy xuất thông tin chính xác. class_id là mã định danh lớp, class_name giúp con người hiểu nội dung dự đoán, còn taxonomy_name xác định hệ thống phân loại đang được sử dụng. Điều này giúp tránh nhầm lẫn khi các model sử dụng những taxonomy khác nhau hoặc có cách đánh số class khác nhau.

6. Vì sao không được lấy `score` của model làm ground truth?

Score chỉ thể hiện mức độ tin cậy của model đối với prediction, không đảm bảo prediction đó chính xác. Trong kết quả thực tế, lớp cab có score 0.510915, nhưng con số này không chứng minh rằng cab là nhãn chuẩn của ảnh. Ground truth phải được xác định dựa trên nội dung ảnh và guideline, không được lấy trực tiếp từ prediction hoặc score của model.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="person" ; `score`=0.912625 ; `bbox_xyxy`=[385.33,69.24,498.92,348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Ta có (x_min, y_min) xác định góc trên bên trái và (x_max, y_max) xác định góc dưới bên phải.Chiều rộng và chiều cao của box lần lượt được tính bằng hiệu tọa độ X và Y tương ứng.
Ví dụ như trên record trên, góc trên trái có tọa độ (385.33,69.24), góc dưới phải có tọa độ (498.92,348.92), rộng 113.58 pixel và cao 279.68 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

Ngưỡng 0.2: có 17 predictions
Ngưỡng 0.35: có 11 vật thể
Ngưỡng 0.6: có 6 predictions

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Cần kiểm tra nhiều bounding box hơn để phát hiện và loại bỏ các box không tương ứng với vật thể thực tế, kiểm tra class bị dự đoán nhầm và xác định những vật thể bị bỏ sót.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Người gán nhãn phải vẽ Bounding Box bao quanh toàn bộ phần vật thể có thể quan sát được, sử dụng hình chữ nhật nhỏ nhất có thể nhưng vẫn chứa đầy đủ vật thể. Box không được bao gồm quá nhiều vùng nền không liên quan. Đối với vật thể bị che khuất, chỉ vẽ theo phần đường biên có thể quan sát được và áp dụng thống nhất quy định về vật thể bị che trong guideline.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Trên ảnh kitchen, có một vật thể bị che khuất một phần nên không thể quan sát đầy đủ đường biên. Tôi lựa chọn đưa trường hợp này lên Lab Coach để xác nhận cách xử lý thay vì tự suy đoán nhãn. Sau khi có hướng dẫn thống nhất, quy tắc xử lý có thể được bổ sung vào guideline để các Annotator áp dụng nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-009" ; `class_name`="person" ; `score`=0.431438 ; số điểm polygon=`polygon_point_count`=65 ; 3–5 điểm đầu của `polygon_xy`=[1.0,264.0],[1.0,309.0],[5.0,309.0],[6.0,308.0],[15.0,308.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Vì vậy, polygon cung cấp thêm thông tin về hình dạng, đường viền và diện tích thực tế của vật thể. Nó mô tả vật thể sát hơn so với box, đặc biệt với những vật thể có hình dạng không vuông vức hoặc nằm gần vật thể khác.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

instance_id là mã định danh riêng cho từng đối tượng được phát hiện trong một ảnh.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Người gán nhãn phải vẽ polygon bám sát đường biên nhìn thấy được của từng vật thể, không bao gồm vùng nền bên ngoài. Khi vật thể bị che khuất, chỉ vẽ đến đường biên còn nhìn thấy, không tự suy đoán hoặc vẽ phần bị che. Các vật thể riêng biệt phải được tạo mask riêng, không gộp chung dù chúng chạm nhau.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Vị trí được chọn: Hai chiếc bát màu hồng ở góc dưới bên trái ảnh, trên mặt bàn
Vấn đề: Hai chiếc bát nằm sát nhau và một phần đường viền bị che khuất, khiến ranh giới giữa hai vật thể không hoàn toàn rõ ràng.
Cách xử lý và lý do:
- Vẽ mask riêng cho từng chiếc bát, không gộp hai chiếc thành một đối tượng.
- Chỉ vẽ theo đường biên thực sự nhìn thấy được.
- Tại vị trí bị chiếc bát kia che khuất, dừng đường biên ở mép phần nhìn thấy; không tự suy đoán phần bị che.
Đây là trường hợp che khuất thông thường. Nếu vẫn nhận biết được hai chiếc bát và xác định được phần đường biên nhìn thấy, người gán nhãn có thể áp dụng quy tắc chung mà không cần escalation. Chỉ escalation nếu không thể xác định đáng tin cậy ranh giới hoặc danh tính của từng vật thể.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một hoặc nhiều nhãn mô tả nội dung toàn ảnh. Ví dụ: kitchen(đơn vị: nhãn ảnh) | Ảnh có nhiều vật thể, dễ nhầm giữa nhãn khái quát và nhãn chi tiết; ví dụ ảnh bếp có người, bát, lò nướng. | Chọn nhãn đúng theo nội dung và taxonomy; không gán nhãn vật thể riêng lẻ nếu chỉ phân loại ảnh. | Nhãn có phản ánh đúng nội dung chính của ảnh không? Có thiếu hoặc thừa nhãn không? |
| Phát hiện vật thể | Mỗi vật thể có class_id, class_name và bounding box bbox_xyxy (đơn vị: pixel). | Ảnh kitchen có nhiều vật thể chạm nhau hoặc bị che. | Vẽ box bao quanh từng vật thể, chọn đúng class; tách riêng các vật thể cùng loại. | Box có bao trọn vật thể không? Có bỏ sót, phát hiện trùng hoặc gán sai class không? |
| Instance segmentation | Mỗi đối tượng có instance_id, class_id, class_name, bbox_xyxy và polygon_xy (đơn vị: pixel). | Ở ảnh kitchen, hai chiếc bát màu hồng phía dưới bên trái nằm sát nhau, biên tiếp xúc khó xác định. | Vẽ polygon riêng cho từng instance, bám theo biên nhìn thấy; không tự vẽ phần bị che. | Kiểm tra mask có sát biên không, có gộp hai vật thể không, instance_id có riêng biệt không? |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Tôi không chia sẻ, sao chép hoặc sử dụng dữ liệu cá nhân ngoài mục đích của bài lab và luôn tuân thủ quy định bảo mật.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Tôi sẽ dừng xử lý, không tự ý sử dụng và báo ngay cho Reviewer/QC để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
