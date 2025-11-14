"""
Test Vietnamese filtering functionality
"""

import yaml
from clean_and_filter import DatasetCleaner


def test_vietnamese_filtering():
    """Test Vietnamese spam and content filtering"""

    # Load config
    with open('datasets_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create cleaner
    cleaner = DatasetCleaner(config)

    # Vietnamese test cases
    test_cases = [
        {
            'name': 'Vietnamese programming content',
            'text': """
            Lập trình Python là một kỹ năng quan trọng trong khoa học dữ liệu.
            Các thuật toán machine learning có thể được triển khai dễ dàng với Python.
            Chúng ta sẽ học về các hàm, biến, và cấu trúc dữ liệu cơ bản.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese science content',
            'text': """
            Nghiên cứu khoa học về trí tuệ nhân tạo đang phát triển rất nhanh.
            Các nhà nghiên cứu đã phát triển nhiều thuật toán học máy mới.
            Phân tích dữ liệu lớn giúp cải thiện độ chính xác của mô hình.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese technology article',
            'text': """
            Công nghệ thông tin đóng vai trò quan trọng trong kỷ nguyên số.
            Phần mềm ứng dụng được phát triển bằng nhiều ngôn ngữ lập trình khác nhau.
            Cơ sở dữ liệu và API là những thành phần quan trọng của hệ thống.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese QA content',
            'text': """
            Câu hỏi: Thuật toán sắp xếp nào có độ phức tạp O(n log n)?
            Câu trả lời: Merge Sort và Quick Sort đều có độ phức tạp O(n log n).
            Giải thích: Cả hai thuật toán đều sử dụng kỹ thuật chia để trị.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese spam - shopping',
            'text': """
            MUA NGAY!!! Giảm giá sốc 90%!!!
            Đăng ký ngay hôm nay để nhận ưu đãi có hạn!!!
            Gọi ngay: 0123456789. Khuyến mãi đặc biệt!!!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Vietnamese spam - get rich quick',
            'text': """
            Kiếm tiền nhanh chỉ trong 7 ngày!!!
            Bí quyết làm giàu thần kỳ!!!
            Thu nhập cao, làm việc tại nhà!!!
            Nhấp vào đây để biết thêm chi tiết!!!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Vietnamese spam - gambling',
            'text': """
            Casino online uy tín!!!
            Cá độ bóng đá, xổ số, cờ bạc!!!
            Trúng thưởng lớn, nhận quà miễn phí!!!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Vietnamese spam - scam',
            'text': """
            Cảnh báo lừa đảo!!! Chiêu trò mạo danh!!!
            Hàng giả, hàng nhái giá rẻ!!!
            Tin nhắn rác, spam liên tục!!!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Vietnamese educational content',
            'text': """
            Hướng dẫn học lập trình cho người mới bắt đầu.
            Tài liệu giáo dục về khoa học máy tính.
            Kiến thức cơ bản về cấu trúc dữ liệu và giải thuật.
            Học tập và nghiên cứu về công nghệ mới.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Mixed English-Vietnamese tech',
            'text': """
            Machine learning và deep learning đang được ứng dụng rộng rãi.
            Các framework như TensorFlow và PyTorch rất phổ biến.
            Chúng ta có thể sử dụng Python để xây dựng các mô hình AI.
            Dữ liệu training phải được chuẩn bị kỹ lưỡng.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese with excessive punctuation',
            'text': """
            Wow!!!!!! Amazing!!!!!
            Tuyệt vời quá!!!!!
            Mua ngay đi!!!!!!
            """,
            'expected': 'FILTERED'
        }
    ]

    print("="*80)
    print("TESTING VIETNAMESE FILTERING")
    print("="*80)
    print()

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 80)

        text = test_case['text']
        expected = test_case['expected']

        result = cleaner.process_text(text)
        actual = 'KEPT' if result else 'FILTERED'

        # Check result
        status = "✓ PASS" if actual == expected else "✗ FAIL"
        if actual == expected:
            passed += 1
        else:
            failed += 1

        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Status: {status}")

        if result:
            print(f"\nCleaned text preview:")
            print(result[:150] + ("..." if len(result) > 150 else ""))

        print()

    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/len(test_cases)*100:.1f}%")
    print()

    # Print overall statistics
    cleaner.print_stats()

    return failed == 0


if __name__ == "__main__":
    success = test_vietnamese_filtering()
    exit(0 if success else 1)
