from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType, IntegerType


def merge_schemas(schema_a, schema_b):
    """
    Merge schema_a into schema_b, adding fields from schema_a that don't exist in schema_b
    (case-insensitive comparison). Handles nested StructType, ArrayType, and MapType.
    Returns a new schema with merged fields.
    """

    def find_field_case_insensitive(field_name, fields):
        """Check if a field exists in a list of fields (case-insensitive)."""
        field_name_lower = field_name.lower()
        for field in fields:
            if field.name.lower() == field_name_lower:
                return field
        return None

    def merge_struct_fields(fields_a, fields_b):
        """Merge fields from two structs, handling nested types."""
        new_fields = fields_b[:]  # Start with schema_b fields

        for field_a in fields_a:
            existing_field = find_field_case_insensitive(field_a.name, new_fields)

            if existing_field is None:
                # Field doesn't exist in schema_b, add it
                new_fields.append(field_a)
            else:
                # Field exists, check if we need to merge nested structures
                if isinstance(field_a.dataType, StructType) and isinstance(existing_field.dataType, StructType):
                    # Merge nested structs
                    merged_struct = merge_schemas(field_a.dataType, existing_field.dataType)
                    new_fields[new_fields.index(existing_field)] = StructField(
                        existing_field.name, merged_struct, existing_field.nullable, existing_field.metadata
                    )
                elif isinstance(field_a.dataType, ArrayType) and isinstance(existing_field.dataType, ArrayType):
                    # Handle array of structs
                    if isinstance(field_a.dataType.elementType, StructType) and isinstance(
                            existing_field.dataType.elementType, StructType):
                        merged_element_type = merge_schemas(field_a.dataType.elementType,
                                                            existing_field.dataType.elementType)
                        new_fields[new_fields.index(existing_field)] = StructField(
                            existing_field.name,
                            ArrayType(merged_element_type, field_a.dataType.containsNull),
                            existing_field.nullable,
                            existing_field.metadata
                        )
                elif isinstance(field_a.dataType, MapType) and isinstance(existing_field.dataType, MapType):
                    # Handle map with struct values
                    if isinstance(field_a.dataType.valueType, StructType) and isinstance(
                            existing_field.dataType.valueType, StructType):
                        merged_value_type = merge_schemas(field_a.dataType.valueType, existing_field.dataType.valueType)
                        new_fields[new_fields.index(existing_field)] = StructField(
                            existing_field.name,
                            MapType(field_a.dataType.keyType, merged_value_type, field_a.dataType.valueContainsNull),
                            existing_field.nullable,
                            existing_field.metadata
                        )

        return StructType(new_fields)

    if not (isinstance(schema_a, StructType) and isinstance(schema_b, StructType)):
        raise ValueError("Both schemas must be StructType")

    return merge_struct_fields(schema_a.fields, schema_b.fields)


# Example usage
def test_merge_schemas():
    # Schema A
    schema_a = StructType([
        StructField("CustomerId", StringType(), True),
        StructField("Personal Info", StructType([
            StructField("Home Address", StringType(), True),
            StructField("Phone Number", StringType(), True),
            StructField("Emergency Contact", StructType([
                StructField("Contact Name", StringType(), True)
            ]), True)
        ]), True),
        StructField("Test Scores", ArrayType(
            StructType([
                StructField("Subject Name", StringType(), True)
            ])
        ), True),
        StructField("Metadata Map", MapType(
            StringType(),
            StructType([
                StructField("Tag Name", StringType(), True)
            ])
        ), True),
        StructField("New Field", IntegerType(), True)
    ])

    # Schema B
    schema_b = StructType([
        StructField("customerid", StringType(), True),
        StructField("personal info", StructType([
            StructField("home address", StringType(), True),
            StructField("email address", StringType(), True)
        ]), True),
        StructField("test scores", ArrayType(
            StructType([
                StructField("subject name", StringType(), True),
                StructField("score value", IntegerType(), True)
            ])
        ), True),
        StructField("metadata map", MapType(
            StringType(),
            StructType([
                StructField("tag value", StringType(), True)
            ])
        ), True)
    ])

    # Merge schemas
    merged_schema = merge_schemas(schema_a, schema_b)

    # Print results
    def print_schema(schema, indent=0, title="Schema"):
        print(f"\n{title}:")
        for field in schema.fields:
            print("  " * indent + f"Field: {field.name}, Type: {field.dataType}")
            if isinstance(field.dataType, StructType):
                print_schema(field.dataType, indent + 1, "Nested Struct")
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                print("  " * (indent + 1) + "Array element:")
                print_schema(field.dataType.elementType, indent + 2, "Array Struct")
            elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
                print("  " * (indent + 1) + "Map value:")
                print_schema(field.dataType.valueType, indent + 2, "Map Struct")

    print_schema(schema_a, title="Schema A")
    print_schema(schema_b, title="Schema B")
    print_schema(merged_schema, title="Merged Schema")


# Run test
test_merge_schemas()
