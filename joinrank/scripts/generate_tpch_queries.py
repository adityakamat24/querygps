#!/usr/bin/env python3
"""
Generate diverse join-heavy queries for TPC-H
"""
import random
import os
from pathlib import Path

random.seed(42)

# TPC-H join patterns
QUERY_TEMPLATES = [
    # 3-way joins
    {
        "name": "q_customer_orders_lineitem",
        "tables": ["customer", "orders", "lineitem"],
        "joins": [
            "customer.c_custkey = orders.o_custkey",
            "orders.o_orderkey = lineitem.l_orderkey"
        ],
        "predicates": [
            ("customer.c_mktsegment = '{segment}'", ["BUILDING", "AUTOMOBILE", "MACHINERY"]),
            ("orders.o_orderstatus = '{status}'", ["F", "O", "P"]),
            ("lineitem.l_shipdate < DATE '{date}'", ["1995-01-01", "1996-01-01", "1997-01-01"])
        ]
    },
    # 4-way joins
    {
        "name": "q_supplier_nation_lineitem_orders",
        "tables": ["supplier", "nation", "lineitem", "orders"],
        "joins": [
            "supplier.s_suppkey = lineitem.l_suppkey",
            "supplier.s_nationkey = nation.n_nationkey",
            "lineitem.l_orderkey = orders.o_orderkey"
        ],
        "predicates": [
            ("nation.n_name = '{nation}'", ["FRANCE", "GERMANY", "UNITED KINGDOM"]),
            ("orders.o_orderdate >= DATE '{date}'", ["1994-01-01", "1995-01-01", "1996-01-01"]),
            ("lineitem.l_quantity > {qty}", ["10", "20", "30"])
        ]
    },
    # 5-way joins
    {
        "name": "q_part_partsupp_supplier_nation_region",
        "tables": ["part", "partsupp", "supplier", "nation", "region"],
        "joins": [
            "part.p_partkey = partsupp.ps_partkey",
            "partsupp.ps_suppkey = supplier.s_suppkey",
            "supplier.s_nationkey = nation.n_nationkey",
            "nation.n_regionkey = region.r_regionkey"
        ],
        "predicates": [
            ("part.p_size = {size}", ["15", "25", "35"]),
            ("part.p_type LIKE '%{type}%'", ["BRASS", "STEEL", "COPPER"]),
            ("region.r_name = '{region}'", ["EUROPE", "ASIA", "AMERICA"])
        ]
    },
    # 6-way joins
    {
        "name": "q_lineitem_orders_customer_nation_supplier_part",
        "tables": ["lineitem", "orders", "customer", "nation", "supplier", "part"],
        "joins": [
            "lineitem.l_orderkey = orders.o_orderkey",
            "orders.o_custkey = customer.c_custkey",
            "customer.c_nationkey = nation.n_nationkey",
            "lineitem.l_suppkey = supplier.s_suppkey",
            "lineitem.l_partkey = part.p_partkey"
        ],
        "predicates": [
            ("orders.o_orderdate BETWEEN DATE '{date1}' AND DATE '{date2}'", 
             [("1995-01-01", "1995-12-31"), ("1996-01-01", "1996-12-31")]),
            ("part.p_size < {size}", ["20", "30"]),
            ("lineitem.l_discount BETWEEN {d1} AND {d2}", [("0.05", "0.07"), ("0.06", "0.08")])
        ]
    },
    # Star join pattern
    {
        "name": "q_lineitem_star",
        "tables": ["lineitem", "part", "supplier", "partsupp", "orders"],
        "joins": [
            "lineitem.l_partkey = part.p_partkey",
            "lineitem.l_suppkey = supplier.s_suppkey",
            "lineitem.l_partkey = partsupp.ps_partkey",
            "lineitem.l_suppkey = partsupp.ps_suppkey",
            "lineitem.l_orderkey = orders.o_orderkey"
        ],
        "predicates": [
            ("part.p_brand = '{brand}'", ["Brand#12", "Brand#23", "Brand#34"]),
            ("supplier.s_acctbal > {bal}", ["5000", "7000"]),
            ("orders.o_totalprice > {price}", ["100000", "200000"])
        ]
    }
]

def generate_query(template, variation_id):
    """Generate a query from template with random predicate values"""
    tables = template["tables"]
    joins = template["joins"]
    
    # Select random predicate values
    where_clauses = []
    for pred_template, values in template["predicates"]:
        value = random.choice(values)
        if isinstance(value, tuple):
            # Handle multiple placeholders
            pred = pred_template
            for i, v in enumerate(value, 1):
                pred = pred.replace(f'{{date{i}}}', v).replace(f'{{d{i}}}', v)
            where_clauses.append(pred)
        else:
            # Single placeholder
            for key in ['{segment}', '{status}', '{date}', '{nation}', 
                       '{qty}', '{size}', '{type}', '{region}', '{brand}', 
                       '{bal}', '{price}']:
                pred_template = pred_template.replace(key, value)
            where_clauses.append(pred_template)
    
    # Build query
    query = f"SELECT COUNT(*)\nFROM {', '.join(tables)}\nWHERE\n  "
    query += " AND\n  ".join(joins)
    if where_clauses:
        query += " AND\n  " + " AND\n  ".join(where_clauses)
    query += ";"
    
    return query

def main():
    output_dir = Path("data/tpch/queries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    queries = []
    query_id = 0
    
    # Generate multiple variations of each template
    for template in QUERY_TEMPLATES:
        for var in range(5):  # 5 variations per template
            query_id += 1
            query_sql = generate_query(template, var)
            query_name = f"{template['name']}_v{var}"
            
            queries.append({
                "id": query_id,
                "name": query_name,
                "sql": query_sql,
                "num_tables": len(template["tables"])
            })
            
            # Save individual query file
            query_file = output_dir / f"q{query_id:03d}_{query_name}.sql"
            with open(query_file, 'w') as f:
                f.write(f"-- Query {query_id}: {query_name}\n")
                f.write(f"-- Tables: {len(template['tables'])}\n\n")
                f.write(query_sql)
    
    # Save metadata
    import json
    metadata_file = output_dir / "queries_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(queries, f, indent=2)
    
    print(f"Generated {len(queries)} queries in {output_dir}")
    print(f"Query distribution:")
    for n_tables in range(3, 7):
        count = sum(1 for q in queries if q["num_tables"] == n_tables)
        print(f"  {n_tables}-way joins: {count}")

if __name__ == "__main__":
    main()