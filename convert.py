import json
from typing import Dict, List, Union, Tuple


def parse_glue_type(glue_type: str) -> Tuple[str, Union[Dict, List, str]]:
    """Parse Glue type string and return type and parameters."""
    glue_type = glue_type.lower().strip()

    if glue_type.startswith('struct<'):
        fields = []
        # Remove struct<> and split fields
        field_str = glue_type[7:-1]
        current_field = ""
        bracket_count = 0  # Tracks < >
        paren_count = 0  # Tracks ( )
        for char in field_str:
            if char == ',' and bracket_count == 0 and paren_count == 0:
                name, dtype = current_field.split(':', 1)
                fields.append((name.strip(), dtype.strip()))
                current_field = ""
            else:
                current_field += char
                if char == '<':
                    bracket_count += 1
                elif char == '>':
                    bracket_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
        if current_field:
            name, dtype = current_field.split(':', 1)
            fields.append((name.strip(), dtype.strip()))
        return 'record', fields

    elif glue_type.startswith('array<'):
        return 'array', glue_type[6:-1]

    elif glue_type.startswith('map<'):
        key_type, value_type = glue_type[4:-1].split(',', 1)
        return 'map', (key_type.strip(), value_type.strip())

    elif glue_type.startswith('decimal('):
        # Extract precision and scale from decimal(p,s)
        params = glue_type[8:-1].split(',')
        if len(params) != 2:
            raise ValueError(f"Invalid decimal format: {glue_type}")
        precision = int(params[0].strip())
        scale = int(params[1].strip())
        return 'decimal', {'precision': precision, 'scale': scale}

    elif glue_type in ('timestamp', 'date'):
        return glue_type, {}

    # Simple types
    return glue_type, {}


def glue_to_avro_schema(glue_schema: str, schema_name: str = "Record") -> Dict:
    """
    Convert Glue schema string to Avro schema with field IDs and optional fields.

    Args:
        glue_schema: Glue schema string
        schema_name: Name of the root schema

    Returns:
        Avro schema dictionary
    """

    def convert_type(glue_type: str, namespace: str = "", field_id: int = 0) -> Tuple[Dict, int]:
        type_name, params = parse_glue_type(glue_type)
        avro_type = {}

        if type_name == 'record':
            fields = []
            current_id = field_id
            for field_name, field_type in params:
                field_schema, current_id = convert_type(field_type, namespace, current_id + 1)
                fields.append({
                    'name': field_name,
                    'type': ['null', field_schema],
                    'default': None,
                    'field-id': current_id
                })
            avro_type = {
                'type': 'record',
                'name': namespace or schema_name,
                'fields': fields
            }

        elif type_name == 'array':
            items_schema, new_id = convert_type(params, namespace, field_id + 1)
            avro_type = {
                'type': 'array',
                'items': items_schema,
                'field-id': field_id
            }

        elif type_name == 'map':
            key_type, value_type = params
            value_schema, new_id = convert_type(value_type, namespace, field_id + 1)
            avro_type = {
                'type': 'map',
                'values': value_schema,
                'field-id': field_id
            }

        elif type_name == 'decimal':
            avro_type = {
                'type': 'bytes',
                'logicalType': 'decimal',
                'precision': params['precision'],
                'scale': params['scale'],
                'field-id': field_id
            }

        elif type_name == 'timestamp':
            avro_type = {
                'type': 'long',
                'logicalType': 'timestamp-millis',
                'field-id': field_id
            }

        elif type_name == 'date':
            avro_type = {
                'type': 'int',
                'logicalType': 'date',
                'field-id': field_id
            }

        else:
            # Map Glue simple types to Avro types
            type_mapping = {
                'string': 'string',
                'int': 'int',
                'integer': 'int',
                'bigint': 'long',
                'double': 'double',
                'float': 'float',
                'boolean': 'boolean',
                'binary': 'bytes'
            }
            avro_type = {
                'type': type_mapping.get(type_name, 'string'),
                'field-id': field_id
            }

        return avro_type, field_id

    # Parse the root schema
    avro_schema, _ = convert_type(glue_schema, schema_name)

    # Add namespace and ensure root is a record
    avro_schema['namespace'] = 'example'
    if avro_schema['type'] != 'record':
        avro_schema = {
            'type': 'record',
            'name': schema_name,
            'namespace': 'example',
            'fields': [{
                'name': 'value',
                'type': ['null', avro_schema],
                'default': None,
                'field-id': 1
            }]
        }

    return avro_schema


# Example usage
def main():
    # Example Glue schema
    glue_schema = """
    struct<
        id:int,
        name:string,
        details:struct<
            age:int,
            scores:array<int>,
            preferences:map<string,struct<code:string,value:double>>,
            salary:decimal(10,2)
        >,
        created:timestamp,
        birth_date:date
    >
    """

    avro_schema = glue_to_avro_schema(glue_schema, "ExampleRecord")

    # Pretty print the result
    print(json.dumps(avro_schema, indent=2))


if __name__ == "__main__":
    main()
