__version__ = '0.1.0-alpha.1'
short_version = __version__


def parse_version_info(version_str):
    """Parse a version string into a tuple.

    Args:
        version_str (str): The version string.
    Returns:
        tuple[int | str]: The version info, e.g., "1.3.0" is parsed into
            (1, 3, 0), and "2.0.0rc1" is parsed into (2, 0, 0, 'rc1').
    """
    _version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            _version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            _version_info.append(int(patch_version[0]))
            _version_info.append(f'rc{patch_version[1]}')
    return tuple(_version_info)


version_info = parse_version_info(__version__)