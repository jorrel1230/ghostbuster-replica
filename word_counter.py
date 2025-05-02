import os
import glob
import argparse

def count_words(directory):
    """Counts the total number of words in all text files within a directory.

    Args:
        directory (str): The path to the directory containing the text files.

    Returns:
        int: The total number of words in all text files.
    """
    total_words = 0
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            words = text.split()
            total_words += len(words)
    return total_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count words in text files within a directory.")
    parser.add_argument("--dir", required=True, help="The directory containing the text files.")
    args = parser.parse_args()

    directory = args.dir

    try:
        word_count = count_words(directory)
        print(f"Total word count in directory '{directory}': {word_count}")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

