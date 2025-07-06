#!/usr/bin/env python3
"""
Quick script to view .kviz preset files in plain text (pretty-printed JSON or dict).
Usage: python kviz_viewer.py path/to/preset.kviz
"""
import sys
import struct
import gzip
import pickle
import json
from pathlib import Path

KVIZ_MAGIC = b'KVIZ'
KVIZ_VERSION = 1
COMPRESSION_NONE = 0
COMPRESSION_GZIP = 1
COMPRESSION_PICKLE = 2

# Data type identifiers for compact encoding
TYPE_FLOAT = 0x01
TYPE_INT = 0x02
TYPE_BOOL = 0x03
TYPE_STRING = 0x04
TYPE_LIST = 0x05
TYPE_DICT = 0x06

def decode_compact_data(data: bytes, offset: int = 0):
    if offset >= len(data):
        raise ValueError("Unexpected end of data")
    data_type = data[offset]
    offset += 1
    if data_type == TYPE_FLOAT:
        value = struct.unpack_from('f', data, offset)[0]
        return value, offset + 4
    elif data_type == TYPE_INT:
        value = struct.unpack_from('i', data, offset)[0]
        return value, offset + 4
    elif data_type == TYPE_BOOL:
        value = struct.unpack_from('?', data, offset)[0]
        return value, offset + 1
    elif data_type == TYPE_STRING:
        length = struct.unpack_from('I', data, offset)[0]
        offset += 4
        value = data[offset:offset + length].decode('utf-8')
        return value, offset + length
    elif data_type == TYPE_LIST:
        length = struct.unpack_from('I', data, offset)[0]
        offset += 4
        result = []
        for _ in range(length):
            item, offset = decode_compact_data(data, offset)
            result.append(item)
        return result, offset
    elif data_type == TYPE_DICT:
        length = struct.unpack_from('I', data, offset)[0]
        offset += 4
        result = {}
        for _ in range(length):
            key, offset = decode_compact_data(data, offset)
            value, offset = decode_compact_data(data, offset)
            result[key] = value
        return result, offset
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def load_kviz_format(filepath: Path):
    with open(filepath, 'rb') as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError("Invalid file header")
        magic, version, compression = struct.unpack('4sII', header)
        if magic != KVIZ_MAGIC:
            raise ValueError(f"Invalid magic number: {magic}")
        if version != KVIZ_VERSION:
            print(f"Warning: Preset version {version} may not be fully compatible")
        data_length = struct.unpack('I', f.read(4))[0]
        data = f.read(data_length)
        if len(data) != data_length:
            raise ValueError("Incomplete data")
        if compression == COMPRESSION_NONE:
            preset_dict, _ = decode_compact_data(data)
        elif compression == COMPRESSION_GZIP:
            json_data = gzip.decompress(data).decode('utf-8')
            preset_dict = json.loads(json_data)
        elif compression == COMPRESSION_PICKLE:
            pickle_data = gzip.decompress(data)
            preset_dict = pickle.loads(pickle_data)
        else:
            raise ValueError(f"Unknown compression level: {compression}")
        return preset_dict

def main():
    if len(sys.argv) != 2:
        print("Usage: python kviz_viewer.py path/to/preset.kviz")
        sys.exit(1)
    kviz_path = Path(sys.argv[1])
    if not kviz_path.exists():
        print(f"File not found: {kviz_path}")
        sys.exit(1)
    try:
        preset = load_kviz_format(kviz_path)
        print(json.dumps(preset, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error reading preset: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
