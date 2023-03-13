"""top level module containing C4.5 parser and Feature abstrctions"""

import os
import re
import sys
import typing
from dataclasses import astuple, dataclass, field
from enum import Enum, IntEnum, unique
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__author__ = "Robbie Dozier"
__email__ = "robbied@case.edu"


@unique
class FeatureType(Enum):
    """
    Enumerate types of features
    """

    #: true/false, 1 or 0, etc
    BINARY = "BINARY"
    #: finite, enumerated set of values, e.g. `['RED', 'GREEN', 'BLUE']`
    NOMINAL = "NOMINAL"
    #: continuous range of values, e.g. real numbers
    CONTINUOUS = "CONTINUOUS"


@dataclass(frozen=True, order=True)
class Feature(object):
    """
    Immutable dataclass representing a feature. Features can be one of the following types: `CLASS`, `ID`, `BINARY`, `NOMINAL`,
    `CONTINUOUS`. This type is immutable and therefore is hashable. It also supports ordering.

    Examples:
        >>> height = Feature('height', FeatureType.CONTINUOUS)

        >>> color = Feature(name='color', ftype=FeatureType.NOMINAL, nominal_values=['RED', 'GREEN', 'BLUE']))
    """

    #: name of the feature
    name: str
    #: type of the feature
    ftype: Union[FeatureType, str]
    #: for nominal features, the possible values the feature can take. Must be a sequence of strings without duplicates.
    nominal_values: Optional[Sequence[str]] = None
    #: for nominal features, the Enum class which dictates the possible features that the feature can attain
    values: Optional[typing.Type[Enum]] = field(init=False, default=None)

    def __post_init__(self):
        # Cast to Type instance
        if type(self.ftype) is not FeatureType:
            object.__setattr__(self, "ftype", FeatureType[self.ftype])

        # Ensure values is appropriately defined
        if self.nominal_values is not None and self.ftype != FeatureType.NOMINAL:
            raise ValueError("values field should only be defined for NOMINAL features")
        elif self.nominal_values is None and self.ftype == FeatureType.NOMINAL:
            raise ValueError(f"missing values field for NOMINAL feature {self.name}")

        # Set nominal feature values
        if self.nominal_values is not None:
            values = IntEnum(self.name, self.nominal_values)
            object.__setattr__(self, "values", values)
            object.__delattr__(self, "nominal_values")

    def __repr__(self):
        if self.values is not None:
            return f"Feature(name='{self.name}', ftype={str(self.ftype)}, values={str(list(self.values))})"
        else:
            return f"Feature(name='{self.name}', ftype={str(self.ftype)})"

    def __hash__(self):
        hash(astuple(self))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Feature):
            return False
        if self.ftype != other.ftype:
            return False

        if self.ftype == FeatureType.NOMINAL:
            return set(self.values) == set(other.values) and self.name == other.name
        else:
            return self.name == other.name

    def to_float(self, value: Any) -> float:
        """
        Converts a value in some other data type to a float.

        Args:
            value (Any): the value to be converted.

        Returns:
            float: The value, represented as a float.

        Examples:
            >>> binary_feature.to_float(True)
            1.

            >>> list(color_nominal_feature.values)
            [<color.RED: 1>, <color.GREEN: 2>, <color.BLUE: 3>]

            >>> color_nominal_feature.to_float(color_nominal_feature.values['GREEN'])
            2.
        """
        if self.ftype == FeatureType.BINARY:
            if value:
                return 1.0
            else:
                return 0.0
        elif self.values is None:
            return float(value)
        elif isinstance(value, str):
            return float(self.values[value].value)
        elif isinstance(value, int):
            return float(self.values(value).value)
        elif isinstance(value, Enum):
            return float(value.value)
        else:
            raise ValueError(
                f"Could not convert {self.name} feature value {value} to float"
            )

    def from_float(self, value: float) -> Union[float, Enum, bool, int]:
        """
        Essentially the reverse of ``to_float()``. Takes a float value and attempts to coerce it into a more useful
         type. This method is provided for convenience but it is not recommended that you rely on it.

        Args:
            value (float): Float value to be converted.

        Returns:
            Union[float, Enum, bool, int]: the original value for a `CONTINUOUS` feature; A boolean for a `BINARY` feature; and int for a `CLASS` or `INDEX` feature; an Enum (or IntEnum) for a NOMINAL feature.

        Examples:
            >>> binary_feature.from_float(1.)
            True

            >>> list(color_nominal_feature.values)
            [<color.RED: 1>, <color.GREEN: 2>, <color.BLUE: 3>]

            >>> color_nominal_feature.from_float(2.)
            <color.GREEN: 2>
        """

        if self.ftype == FeatureType.CONTINUOUS:
            return value
        elif self.ftype == FeatureType.BINARY:
            return bool(value)
        else:
            return self.values(int(value))


def nominal_str_to_numeric(feature: Feature, arr: Iterable[str]) -> np.ndarray:
    """
    Takes an iterable of strings with nominal feature names and converts them to ints.
    Args:
        feature (Feature): `Feature` to convert. Must be a nominal feature (`values` attribute is not `None` and is an `IntEnum`)
        arr (Iterable[str]): Iterable of strings. All elements must be valid values or None. None and invalid values will map to np.nan

    Returns:
        np.ndarray: A numpy array of floats corresponding to the indices of nominal values

    Examples:
        >>> nominal_str_to_numeric(color_feature, ['RED', 'GREEN', 'BLUE'])
        np.ndarray([1.0, 2.0, 3.0], dtype=float)
    """
    if not issubclass(feature.values, IntEnum):
        raise AttributeError(
            f"feature.values must be type IntEnum, not {type(feature.values)}"
        )

    # It would be nice to have a way to vectorize this but I'm not sure how
    indices = []
    for name in arr:
        try:
            indices.append(feature.values[str(name)].value)
        except KeyError:
            indices.append(None)

    return np.array(indices, dtype=float)


def nominal_numeric_to_str(feature: Feature, arr: np.ndarray) -> Iterable:
    """
    Takes an iterable of ints of nominal feature values and converts them to their str names.

    Args:
        feature (Feature): `Feature` to convert. Must be a nominal feature (`values` attribute is not `None` and is an `IntEnum`)
        arr (np.ndarray): Numpy array of valid values of the nominal feature.

    Returns:
        np.ndarray: A numpy array of strings corresponding to the names of nominal features

    Examples:
        >>> nominal_numeric_to_str(color_feature, np.ndarray([1, 2, 3]))
        np.ndarray(['RED', 'GREEN', 'BLUE'], dtype=object)
    """

    if not issubclass(feature.values, IntEnum):
        raise AttributeError(
            f"feature.values must be type IntEnum, not {type(feature.values)}"
        )

    # Prepend None because Enum flags start at 1
    feature_arr = np.array([None] + [f.name for f in feature.values], dtype="object")

    return feature_arr[arr.astype(int)]


# Beyond here is code for parsing C45 data

_NAMES_EXT = ".names"
_DATA_EXT = ".data"

_COMMENT_RE = "#.*"
_BINARY_RE = "\\s*0\\s*,\\s*1\\s*"


def parse_c45(
    file_base: str, root_dir: str = "."
) -> Tuple[List[Feature], np.ndarray, np.ndarray]:
    """returns a schema, example ndarray, and label ndarray from a C4.5-formatted data file.
    note that this function will check in `root_dir` and all descendent directories

    Args:
        file_base (str): base filename to be parsed, e.g. `file_base.names`
        root_dir (str, optional): root directory to start recursively searching for file_base. Defaults to ".".

    Returns:
        Tuple[List[Feature], np.ndarray, np.ndarray]: tuple containing the schema, the data, and the labels. The schema will be a list of Feature instances, the
        data will be a float numpy array of shape (n_examples, n_features), and the labels will be an int numpy array
        of shape (n_examples,)

    Examples:

        Note that these are equivalent if the `spam/` dir appears inside of `440data/`.
        Each returns a 3-tuple of `List[Feature]` and numpy ndarrays for examples and labels:
        >>> schema, X, y = parse_c45("spam", "") # scan current dir and subdirs
        >>> schema, X, y = parse_c45("spam") # equivalent to the 2nd example
        >>> schema, X, y = parse_c45("spam", "440data") # scan 440data/ dir and subdirs

        `schema` encodes the name, type, and possible values for each feature
        >>> schema
        [Feature(name='geoDistance', ftype=<FeatureType.CONTINUOUS>, values=None), ...]

        `X` denotes a 2-dimensional example matrix (capital letters denote matrices, lowercase letters denote vectors).
        Each row is an example, each column is a feature.
        >>> X.shape
        (74736, 19)

        `y` denotes the label vector. It contains the class label for each example.
        >>> y.shape
        (74736,)

        The same example as above using the smaller, `example` dataset:
        >>> schema, X, y = parse_c45("example")
        >>> schema
        [Feature(name='f1', ftype=FeatureType.BINARY), Feature(name='f2', ftype=FeatureType.NOMINAL, values=[<f2.Monday: 1>, <f2.Tuesday: 2>, <f2.Wednesday: 3>, <f2.Thursday: 4>, <f2.Friday: 5>]), Feature(name='f3', ftype=FeatureType.CONTINUOUS), Feature(name='f4', ftype=FeatureType.NOMINAL, values=[<f4.A12: 1>, <f4.A13: 2>, <f4.A14: 3>])]
        >>> X
        [[0.        1.        0.94      1.       ]
         [0.        2.        1.        2.       ]
         [0.        3.        1.5       3.       ]
         [      nan 4.        1.1       3.       ]
         [0.        5.        2.3       1.       ]
         [1.        3.        0.86      2.       ]
         [0.        2.        3.14      1.       ]
         [0.        1.        2.81            nan]
         [1.        4.        0.9932456 2.       ]
         [1.        5.        2.        1.       ]]
        >>> y
        [0 0 0 1 1 0 1 0 1 1]
    """
    schema_name = file_base + _NAMES_EXT
    schema_filename = _find_file(schema_name, root_dir)
    if schema_filename is None:
        raise ValueError("Schema file not found")

    data_name = file_base + _DATA_EXT
    data_filename = _find_file(data_name, root_dir)
    if data_filename is None:
        raise ValueError("Data file not found")

    return _parse_c45(schema_filename, data_filename)


def _parse_c45(
    schema_filename, data_filename
) -> Tuple[List[Feature], np.ndarray, np.ndarray]:
    """Parses C4.5 given file names"""
    try:
        schema = _parse_schema(schema_filename)
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise Exception("Error parsing schema: %s" % e)

    try:
        df = _parse_and_preprocess_csv(schema, data_filename)
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise Exception("Error parsing examples: %s" % e)

    # Remove class labels from schema
    y = df.pop("label")

    return (
        schema,
        df.to_numpy(dtype=float, copy=True),
        y.to_numpy(dtype=int, copy=True),
    )


def _parse_schema(schema_filename) -> List[Feature]:
    """Parses C4.5 '.names' schema file"""
    features = []
    with open(schema_filename) as schema_file:
        for line in schema_file:
            feature = _parse_feature(line)
            if feature is not None:
                features.append(feature)

    return features


def _parse_feature(line) -> Optional[Feature]:
    """
    Parse a feature from the given line.

    """
    line = _trim_line(line)
    if len(line) == 0:
        # Blank line
        return None
    if re.match(_BINARY_RE, line) is not None:
        # Class feature, ignore
        return None
    colon = line.find(":")
    if colon < 0:
        raise Exception("No feature name found.")
    name = line[:colon].strip()
    remainder = line[(colon + 1):]
    values = _parse_values(remainder)
    if name == "index":
        # Index feature, ignore
        return None
    elif len(values) == 1 and values[0].startswith("continuous"):
        return Feature(name, FeatureType.CONTINUOUS)
    elif values == ["0", "1"]:
        return Feature(name, FeatureType.BINARY)
    else:
        return Feature(name, FeatureType.NOMINAL, values)


def _parse_values(value_string):
    """Parse comma-delimited values from a string"""
    values = list()
    for raw in value_string.split(","):
        raw = raw.strip()
        if len(raw) > 1 and raw[0] == '"' and raw[-1] == '"':
            raw = raw[1:-1].strip()
        values.append(raw)
    return values


def _parse_and_preprocess_csv(schema, data_filename) -> pd.DataFrame:
    """Parse examples from a '.data' file given a schema using pandas"""

    df = pd.read_csv(data_filename, header=None, comment="#").replace(
        r'"|\s+', "", regex=True
    )
    df.columns = ["index"] + [f.name for f in schema] + ["label"]
    df.replace("?", np.nan, inplace=True)

    # Conform to schema, cast features to float, classes/indices to int
    for feature in schema:
        if feature.ftype == FeatureType.NOMINAL:
            # Convert to numeric
            df[feature.name] = nominal_str_to_numeric(feature, df[feature.name])
        else:
            # Cast to float
            df[feature.name] = df[feature.name].astype(float)

    df["index"] = df["index"].astype(int)
    df["label"] = df["label"].astype(int)

    df.set_index("index", inplace=True, drop=True)

    return df


def _trim_line(line: str) -> str:
    """Removes comments and periods from the given line"""
    line = re.sub(_COMMENT_RE, "", line)
    line = line.strip()
    if len(line) > 0 and line[-1] == ".":
        line = line[:-1].strip()
    return line


def _find_file(filename: str, rootdir: str) -> str:
    """
    Finds a file with filename located in some
    subdirectory of the root directory
    """
    for dirpath, _, filenames in os.walk(os.path.expanduser(rootdir)):
        if filename in filenames:
            return os.path.join(dirpath, filename)
