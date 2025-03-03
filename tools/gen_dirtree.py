import os
import argparse


def generate_file_tree(start_path, output_file, exclude_dirs=None, exclude_files=None):
    """
    Generate a file tree structure and save it to a txt file

    Args:
        start_path: The root directory path to start traversing
        output_file: Path of the output txt file
        exclude_dirs: List of directory names to exclude (e.g., ['.git', '__pycache__'])
        exclude_files: List of file names to exclude (e.g., ['__init__.py'])
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', '.idea']

    if exclude_files is None:
        exclude_files = ['__init__.py']

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Project Tree for: {os.path.abspath(start_path)}\n")
        f.write("=" * 50 + "\n")

        for root, dirs, files in os.walk(start_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            # Filter out excluded files
            files = [file for file in files if file not in exclude_files]

            # Calculate current level for indentation
            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * (level)

            # Write current directory name
            dirname = os.path.basename(root)
            if root != start_path:
                f.write(f"{indent}├── {dirname}/\n")

            # Write file names with proper indentation
            subindent = '│   ' * (level + 1)
            for file in sorted(files):
                f.write(f"{subindent}├── {file}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate a file tree structure and save it to a txt file')

    parser.add_argument('--project_path', '-p', type=str,
                        default="../mmdepth",
                        help='The root directory path to start traversing')

    parser.add_argument('--output_file', '-o', type=str,
                        default="../project_tree.txt",
                        help='Path of the output txt file')

    parser.add_argument('--exclude_dirs', '-ed', type=str, nargs='+',
                        default=['.git', '__pycache__', 'node_modules', '.idea'],
                        help='List of directory names to exclude')

    parser.add_argument('--exclude_files', '-ef', type=str, nargs='+',
                        default=['__init__.py'],
                        help='List of file names to exclude (default: __init__.py)')

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Generate the file tree
    generate_file_tree(args.project_path, args.output_file, args.exclude_dirs, args.exclude_files)

    print(f"File tree has been generated at: {args.output_file}")