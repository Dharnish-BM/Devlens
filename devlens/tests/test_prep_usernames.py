import os
import pytest
from devlens.data_collection.prep_usernames import parse_and_clean_line, validate_username, process_classmate_links


def test_parse_and_clean_line():
    assert parse_and_clean_line("https://github.com/torvalds")[0] == "torvalds"
    assert parse_and_clean_line("http://github.com/HariniZoe")[0] == "HariniZoe"
    assert parse_and_clean_line("https://devachandran23.github.io/wtlab-blog-ex1/")[0] == "devachandran23"
    assert parse_and_clean_line("gldinesh1211 (gldinesh)")[0] == "gldinesh1211"
    assert parse_and_clean_line("  @octocat  ")[0] == "octocat"
    assert parse_and_clean_line("bareuser")[0] == "bareuser"


def test_validate_username():
    assert validate_username("torvalds")[0] is True
    assert validate_username("HariniZoe")[0] is True
    assert validate_username("topics")[0] is False
    assert validate_username("invalid username")[0] is False
    assert validate_username("-invalid-")[0] is False


def test_process_classmate_links(tmp_path):
    input_file = tmp_path / "links.txt"
    output_file = tmp_path / "usernames.txt"

    input_file.write_text(
        "https://github.com/user1\n"
        "https://github.com/user2\n"
        "https://github.com/user1\n"  # duplicate
        "http://user3.github.io/blog\n"
        "bad url http://example.com/notgithub\n"
    )

    summary = process_classmate_links(input_file=str(input_file), output_file=str(output_file))

    assert summary["total_input_lines"] == 5
    assert summary["valid_unique_usernames"] == 3
    assert summary["duplicates_removed"] == 1
    assert summary["failed_lines_count"] == 1
    assert summary["usernames"] == ["user1", "user2", "user3"]
