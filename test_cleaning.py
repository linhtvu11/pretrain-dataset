"""
Test script for text cleaning and filtering functionality
Run this to verify that the cleaning and filtering works correctly
"""

import yaml
from clean_and_filter import DatasetCleaner


def test_cleaning():
    """Test the cleaning and filtering functionality"""

    # Load config
    with open('datasets_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create cleaner
    cleaner = DatasetCleaner(config)

    # Test cases
    test_cases = [
        {
            'name': 'Valid Python code',
            'text': """
            Here's a simple Python function that calculates factorial:

            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n - 1)

            This is a recursive implementation.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Scientific content',
            'text': """
            Recent research in machine learning has shown that transformer
            architectures can achieve state-of-the-art results on various
            natural language processing tasks. The attention mechanism allows
            the model to focus on relevant parts of the input sequence.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Spam content',
            'text': """
            BUY NOW!!! LIMITED OFFER!!!
            Click here to win $$$$$
            Subscribe now to our newsletter
            WINNER WINNER WINNER!!!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Too short',
            'text': "Hi",
            'expected': 'FILTERED'
        },
        {
            'name': 'HTML content',
            'text': """
            <html><body>
            <h1>Machine Learning Tutorial</h1>
            <p>This tutorial covers basic concepts of machine learning including
            supervised learning, unsupervised learning, and reinforcement learning.</p>
            </body></html>
            """,
            'expected': 'KEPT'  # Should keep after removing HTML tags
        },
        {
            'name': 'Code with explanation',
            'text': """
            The following JavaScript code implements a binary search algorithm:

            function binarySearch(arr, target) {
                let left = 0;
                let right = arr.length - 1;

                while (left <= right) {
                    const mid = Math.floor((left + right) / 2);
                    if (arr[mid] === target) return mid;
                    if (arr[mid] < target) left = mid + 1;
                    else right = mid - 1;
                }
                return -1;
            }

            This algorithm has O(log n) time complexity.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Biology content',
            'text': """
            DNA replication is a fundamental biological process that occurs in all
            living organisms. It involves the synthesis of two identical DNA molecules
            from a single parent DNA molecule. The process is semi-conservative, meaning
            each new DNA molecule consists of one original strand and one newly synthesized strand.
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Casino spam',
            'text': """
            Win big at our online casino! Play now and claim your prize!
            Lottery winner announced! Click here!
            """,
            'expected': 'FILTERED'
        },
        {
            'name': 'Technical documentation',
            'text': """
            REST API Endpoint Documentation

            GET /api/v1/users
            Returns a list of all users in the system.

            Parameters:
            - page (int): Page number for pagination
            - limit (int): Number of items per page

            Response:
            {
                "users": [...],
                "total": 100,
                "page": 1
            }
            """,
            'expected': 'KEPT'
        },
        {
            'name': 'Vietnamese tech content',
            'text': """
            Lập trình Python là một kỹ năng quan trọng trong khoa học dữ liệu.
            Python cung cấp nhiều thư viện mạnh mẽ như NumPy, Pandas, và Scikit-learn
            để xử lý và phân tích dữ liệu. Các thuật toán machine learning có thể
            được triển khai dễ dàng với Python.
            """,
            'expected': 'KEPT'
        }
    ]

    print("="*80)
    print("TESTING TEXT CLEANING AND FILTERING")
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
            print(result[:200] + ("..." if len(result) > 200 else ""))

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
    success = test_cleaning()
    exit(0 if success else 1)
