# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import apache_beam as beam
import argparse
import datetime
import json
import logging
import os
import tensorflow_data_validation as tfdv

from apache_beam.options.pipeline_options import StandardOptions
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='GCS or local directory.')
    parser.add_argument(
        '--csv-data-for-inference',
        type=str,
        required=True,
        help='GCS path of the CSV file from which to infer the schema.')
    parser.add_argument(
        '--csv-data-to-validate',
        type=str,
        help='GCS path of the CSV file whose contents should be validated.')
    parser.add_argument(
        '--column-names',
        type=str,
        help='GCS json file containing a list of column names.')
    parser.add_argument(
        '--key-columns',
        type=str,
        help='Comma separated list of columns to treat as keys.')

    args = parser.parse_args()
    return args


def convert_feature_to_json(feature, key_columns):
    feature_json = {'name': feature.name}
    feature_type = schema_pb2.FeatureType.Name(feature.type)
    if feature.name in key_columns:
        feature_json['type'] = 'KEY'
    elif (feature_type == 'INT' or feature_type == 'FLOAT' or
          feature.HasField('int_domain') or feature.HasField('float_domain')):
        feature_json['type'] = 'NUMBER'
    elif feature.HasField('bool_domain'):
        feature_json['type'] = 'CATEGORY'
    elif feature_type == 'BYTES':
        if (feature.HasField('domain') or
                feature.HasField('string_domain') or
                (feature.HasField('distribution_constraints') and
                 feature.distribution_constraints.min_domain_mass > 0.95)):
            feature_json['type'] = 'CATEGORY'
        else:
            feature_json['type'] = 'TEXT'
    else:
        feature_json['type'] = 'KEY'
    return feature_json


def convert_schema_proto_to_json(schema, column_names, key_columns):
    column_schemas = {}
    for feature in schema.feature:
        column_schemas[feature.name] = (
            convert_feature_to_json(feature, key_columns))
    schema_json = []
    for column_name in column_names:
        schema_json.append(column_schemas[column_name])
    return schema_json


def run_validator(output_dir, column_names, key_columns, csv_data_file,
                  csv_data_file_to_validate):
    """Writes a TFDV-generated schema.
    Args:
      output_dir: output folder
      column_names: list of names for the columns in the CSV file. If omitted,
          the first line is treated as the column names.
      key_columns: list of the names for columns that should be
          treated as unique keys.
      csv_data_file: name of the CSV file to analyze and generate a schema.
      csv_data_file_to_validate: name of a CSV file to validate
          against the schema.
    """
    logging.getLogger().info('running in local mode')
    pipeline_options = None

    logging.getLogger().info('starting stats on tfdv')
    stats = tfdv.generate_statistics_from_csv(
        data_location=csv_data_file,
        column_names=column_names,
        delimiter=',',
        output_path=os.path.join(output_dir, 'data_stats.tfrecord'),
        pipeline_options=pipeline_options)
    schema = tfdv.infer_schema(stats)

    logging.getLogger().info('loading output_schema.pb2')
    with open('{}/output_schema.pb2'.format(output_dir), 'w+') as f:
        f.write(schema.SerializeToString())

    logging.getLogger().info('loading [output_dir] {} schema.pb2'.format(output_dir))
    with file_io.FileIO(os.path.join(output_dir, 'schema.pb2'), 'w+') as f:
        logging.getLogger().info('loading schema to {}'.format(f.name))
        f.write(schema.SerializeToString())

    schema_json = convert_schema_proto_to_json(
        schema, column_names, key_columns)

    logging.getLogger().info(' logging output_schema.json')
    with open('{}/output_schema.json'.format(output_dir), 'w+') as f:
        json.dump(schema_json, f)
    schema_json_file = os.path.join(output_dir, 'schema.json')

    with file_io.FileIO(schema_json_file, 'w+') as f:
        logging.getLogger().info('logging JSON schema to {}'.format(f.name))
        json.dump(schema_json, f)

    with open('{}/schema.txt'.format(output_dir), 'w+') as f:
        logging.getLogger().info('schema.txt to {}'.format(f.name))
        f.write(schema_json_file)

    logging.getLogger().info('Schema Write Done...')

    if not csv_data_file_to_validate:
        logging.getLogger().info('No csv file to validate')
        return

    logging.getLogger().info('Validation Stats...')
    validation_stats = tfdv.generate_statistics_from_csv(
        data_location=csv_data_file_to_validate,
        column_names=column_names,
        delimiter=',',
        output_path=os.path.join(output_dir, 'validation_data_stats.tfrecord'),
        pipeline_options=pipeline_options)
    anomalies = tfdv.validate_statistics(validation_stats, schema)

    logging.getLogger().info('logging output validation results ...')
    with open('{}/output_validation_result.txt'.format(output_dir), 'w+') as f:
        if len(anomalies.anomaly_info.items()) > 0:
            f.write('invalid')
        else:
            f.write('valid')
            return

    # logging.getLogger().info('logging anomalies result ...')
    # with file_io.FileIO(os.path.join(output_dir, 'anomalies.pb2'), 'w+') as f:
    #     logging.getLogger().info('logging anomalies to {}'.format(f.name))
    #     f.write(anomalies.SerializeToString())
    #
    # for feature_name, anomaly_info in anomalies.anomaly_info.items():
    #     logging.getLogger().error(
    #         'Anomaly in feature "{}": {}'.format(
    #             feature_name, anomaly_info.description))
    return 0


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_arguments()
    column_names = None
    if args.column_names:
        column_names = json.loads(
            file_io.read_file_to_string(args.column_names))

    logging.getLogger().info('starting validator ...')
    logging.getLogger().info('args.output: {}'.format(args.output))
    logging.getLogger().info('args.key_columns: {}'.format(args.key_columns))
    logging.getLogger().info('args.csv_data_for_inference: {}'.format(args.csv_data_for_inference))
    logging.getLogger().info('args.csv_data_to_validate: {}'.format(args.csv_data_to_validate))

    run_validator(args.output,
                  column_names,
                  args.key_columns.split(','),
                  args.csv_data_for_inference,
                  args.csv_data_to_validate)


if __name__ == "__main__":
    main()
