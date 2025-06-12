from typing import Union
from iceberg.schema import Schema as IcebergSchema
from iceberg.types import (
    StructType as IcebergStructType, ListType as IcebergListType, MapType as IcebergMapType,
    PrimitiveType, IntegerType as IcebergIntegerType, LongType as IcebergLongType,
    FloatType as IcebergFloatType, DoubleType as IcebergDoubleType, StringType as IcebergStringType,
    BooleanType as IcebergBooleanType, BinaryType as IcebergBinaryType, TimestampType as IcebergTimestampType,
    DateType as IcebergDateType, DecimalType as IcebergDecimalType, UUIDType as IcebergUUIDType
)
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, IntegerType, LongType, FloatType, DoubleType,
    StringType, BooleanType, BinaryType, TimestampType, DateType, DecimalType
)

def iceberg_to_pyspark_schema(iceberg_schema: Union[IcebergSchema, IcebergStructType]) -> StructType:
    """
    Convert an Apache Iceberg schema to a PySpark schema.
    
    Args:
        iceberg_schema: Iceberg schema (Schema or StructType object)
        
    Returns:
        StructType: Equivalent PySpark schema
    """
    def _convert_type(iceberg_type: IcebergStructType) -> Union[StructType, ArrayType, MapType, IntegerType, 
                                                              LongType, FloatType, DoubleType, StringType, 
                                                              BooleanType, BinaryType, TimestampType, 
                                                              DateType, DecimalType]:
        """Convert an Iceberg type to a PySpark type."""
        if isinstance(iceberg_type, IcebergStructType):
            fields = [
                StructField(field.name, _convert_type(field.type), not field.required)
                for field in iceberg_type.fields
            ]
            return StructType(fields)
        elif isinstance(iceberg_type, IcebergListType):
            element_type = _convert_type(iceberg_type.element_type)
            return ArrayType(element_type, not iceberg_type.element_required)
        elif isinstance(iceberg_type, IcebergMapType):
            key_type = _convert_type(iceberg_type.key_type)
            value_type = _convert_type(iceberg_type.value_type)
            return MapType(key_type, value_type, not iceberg_type.value_required)
        elif isinstance(iceberg_type, PrimitiveType):
            type_mapping = {
                IcebergIntegerType: IntegerType(),
                IcebergLongType: LongType(),
                IcebergFloatType: FloatType(),
                IcebergDoubleType: DoubleType(),
                IcebergStringType: StringType(),
                IcebergBooleanType: BooleanType(),
                IcebergBinaryType: BinaryType(),
                IcebergTimestampType: TimestampType(),
                IcebergDateType: DateType(),
                IcebergUUIDType: StringType()  # UUID maps to StringType in PySpark
            }
            if isinstance(iceberg_type, IcebergDecimalType):
                return DecimalType(precision=iceberg_type.precision, scale=iceberg_type.scale)
            return type_mapping.get(type(iceberg_type), StringType())  # Fallback to StringType
        else:
            return StringType()  # Fallback for unhandled types

    # If input is an Iceberg Schema, get its StructType
    if isinstance(iceberg_schema, IcebergSchema):
        iceberg_struct = iceberg_schema.as_struct()
    else:
        iceberg_struct = iceberg_schema

    if not isinstance(iceberg_struct, IcebergStructType):
        raise ValueError("Iceberg schema must be a Schema or StructType")

    return _convert_type(iceberg_struct)
