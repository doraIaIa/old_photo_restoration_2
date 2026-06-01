# Final Report Talking Points

## Câu chuyện chính

- Bài toán không chỉ là inpainting; chất lượng mask quyết định phần lớn kết quả phục hồi.
- r009 synthetic-only gần như không chuyển domain sang ảnh cũ thật.
- r010 real-domain fine-tune cải thiện mạnh trên real test.
- r011 repair-mask fine-tune tiếp tục cải thiện khi nhãn gần hơn với vùng cần repair thực tế.
- External/manual mask là upper bound hoặc diagnosis, không phải automatic pipeline.

## Kết quả nên nhấn mạnh

| Mốc | Ý nghĩa |
|---|---|
| r009 real test IoU 0.002222, F1 0.004434 | Synthetic-only không đủ cho real old-photo cracks. |
| r010 real test IoU 0.292728, F1 0.452884 | Real-domain fine-tune đem lại bước nhảy lớn. |
| r011 repair_v1 test IoU 0.447877, F1 0.618667 | Repair-mask target phù hợp hơn với pipeline restoration. |
| r011 thin GT test IoU 0.371838, F1 0.542102 | Có trade-off giữa mask repair rộng hơn và thin/manual GT. |

## Demo variants

- `auto_r011`: baseline automatic.
- `auto_r011_union`: thêm CV signal để bắt crack mà DL có thể bỏ sót.
- `auto_r011_refined`: dùng Module 1.5 để làm sạch DL mask.
- `auto_r011_union_refined`: enhanced automatic, cần review false positive.
- `auto_r011_union_refined_face_auto`: chỉ nên khuyến nghị nếu visual review xác nhận hữu ích.
- `external` và `external_face_auto`: diagnosis/upper bound.

## Variant automatic khuyến nghị

Nếu visual review chọn nó, variant automatic được khuyến nghị là `auto_r011_union_refined_face_auto` cho bản có Module 3 hoặc `auto_r011_union_refined` cho bản không dùng face restoration.

Trong kết quả r010 trước đó, `r010_union_cv_t070_dilate1` chỉ nên được gọi là automatic recommendation khi visual review xác nhận nó tốt hơn các biến thể khác.

## Hạn chế cần nói rõ

- Mask generation là bottleneck chính.
- Face restoration hiện là dependency-gated wrapper, không phải bằng chứng rằng CodeFormer/GFPGAN đã được áp dụng thành công.
- Manual upper bound chỉ là ceiling/diagnosis.
- Chưa train r012 trong Blueprint 2.1.

## Kết luận ngắn

Fine-tune real-domain giải quyết phần lớn domain gap so với r009 synthetic-only. Tuy vậy, chất lượng phục hồi cuối cùng vẫn bị chặn chủ yếu bởi mask generation, nên các cải tiến tiếp theo nên ưu tiên nhãn repair-mask tốt hơn, visual review có hệ thống và refinement giảm false positive.
