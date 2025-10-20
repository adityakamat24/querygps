# setup_tpch.ps1
# TPC-H Setup Script for Windows (PowerShell)
# Usage: .\setup_tpch.ps1

$ErrorActionPreference = "Stop"

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "TPC-H Setup for Windows" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Configuration
$SCALE_FACTOR = 1
$DB_NAME = "tpch_s1"
$DB_USER = $env:USERNAME
$DB_HOST = "localhost"
$DB_PORT = "5432"

# Directories
$DATA_DIR = "data\tpch"
New-Item -ItemType Directory -Force -Path "$DATA_DIR\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA_DIR\sql" | Out-Null
New-Item -ItemType Directory -Force -Path "$DATA_DIR\queries" | Out-Null

Write-Host ""
Write-Host "Step 1: Downloading TPC-H dbgen..." -ForegroundColor Yellow

# Check if tpch-dbgen exists
if (-Not (Test-Path "tpch-dbgen")) {
    Write-Host "Cloning TPC-H dbgen repository..."
    git clone https://github.com/electrum/tpch-dbgen.git
} else {
    Write-Host "TPC-H dbgen already exists, skipping download."
}

Write-Host ""
Write-Host "Step 2: Getting dbgen executable..." -ForegroundColor Yellow

$dbgenPath = "tpch-dbgen\dbgen.exe"

if (-Not (Test-Path $dbgenPath)) {
    Write-Host "Downloading pre-built dbgen.exe for Windows..."
    $dbgenUrl = "https://github.com/gregrahn/tpch-kit/raw/master/dbgen/dbgen.exe"
    try {
        Invoke-WebRequest -Uri $dbgenUrl -OutFile $dbgenPath
        Write-Host "Downloaded successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Could not download pre-built binary." -ForegroundColor Red
        Write-Host "Please download manually from: https://github.com/gregrahn/tpch-kit/releases" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "dbgen.exe already exists."
}

Write-Host ""
Write-Host "Step 3: Generating TPC-H data (Scale Factor $SCALE_FACTOR)..." -ForegroundColor Yellow

Push-Location "tpch-dbgen"
if (Test-Path "dbgen.exe") {
    Write-Host "Generating data files..."
    .\dbgen.exe -s $SCALE_FACTOR
    
    # Move generated files
    Move-Item -Path "*.tbl" -Destination "..\$DATA_DIR\raw\" -Force
    Write-Host "Data files generated and moved to $DATA_DIR\raw\" -ForegroundColor Green
} else {
    Write-Host "dbgen.exe not found! Please build it first." -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

Write-Host ""
Write-Host "Step 4: Creating PostgreSQL database..." -ForegroundColor Yellow

# Check if psql is available
try {
    $psqlVersion = psql --version
    Write-Host "Found PostgreSQL: $psqlVersion"
} catch {
    Write-Host "PostgreSQL not found in PATH!" -ForegroundColor Red
    Write-Host "Please install PostgreSQL and add it to PATH." -ForegroundColor Yellow
    Write-Host "Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
    exit 1
}

# Drop and create database
Write-Host "Dropping existing database (if any)..."
psql -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>$null

Write-Host "Creating database $DB_NAME..."
psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;"

Write-Host ""
Write-Host "Step 5: Creating tables..." -ForegroundColor Yellow

# Create schema SQL
$schemaSQL = @"
CREATE TABLE nation (
    n_nationkey  INTEGER NOT NULL PRIMARY KEY,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INTEGER NOT NULL,
    n_comment    VARCHAR(152)
);

CREATE TABLE region (
    r_regionkey  INTEGER NOT NULL PRIMARY KEY,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152)
);

CREATE TABLE part (
    p_partkey     INTEGER NOT NULL PRIMARY KEY,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment     VARCHAR(23) NOT NULL
);

CREATE TABLE supplier (
    s_suppkey     INTEGER NOT NULL PRIMARY KEY,
    s_name        CHAR(25) NOT NULL,
    s_address     VARCHAR(40) NOT NULL,
    s_nationkey   INTEGER NOT NULL,
    s_phone       CHAR(15) NOT NULL,
    s_acctbal     DECIMAL(15,2) NOT NULL,
    s_comment     VARCHAR(101) NOT NULL
);

CREATE TABLE partsupp (
    ps_partkey     INTEGER NOT NULL,
    ps_suppkey     INTEGER NOT NULL,
    ps_availqty    INTEGER NOT NULL,
    ps_supplycost  DECIMAL(15,2) NOT NULL,
    ps_comment     VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE customer (
    c_custkey     INTEGER NOT NULL PRIMARY KEY,
    c_name        VARCHAR(25) NOT NULL,
    c_address     VARCHAR(40) NOT NULL,
    c_nationkey   INTEGER NOT NULL,
    c_phone       CHAR(15) NOT NULL,
    c_acctbal     DECIMAL(15,2) NOT NULL,
    c_mktsegment  CHAR(10) NOT NULL,
    c_comment     VARCHAR(117) NOT NULL
);

CREATE TABLE orders (
    o_orderkey       INTEGER NOT NULL PRIMARY KEY,
    o_custkey        INTEGER NOT NULL,
    o_orderstatus    CHAR(1) NOT NULL,
    o_totalprice     DECIMAL(15,2) NOT NULL,
    o_orderdate      DATE NOT NULL,
    o_orderpriority  CHAR(15) NOT NULL,
    o_clerk          CHAR(15) NOT NULL,
    o_shippriority   INTEGER NOT NULL,
    o_comment        VARCHAR(79) NOT NULL
);

CREATE TABLE lineitem (
    l_orderkey      INTEGER NOT NULL,
    l_partkey       INTEGER NOT NULL,
    l_suppkey       INTEGER NOT NULL,
    l_linenumber    INTEGER NOT NULL,
    l_quantity      DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount      DECIMAL(15,2) NOT NULL,
    l_tax           DECIMAL(15,2) NOT NULL,
    l_returnflag    CHAR(1) NOT NULL,
    l_linestatus    CHAR(1) NOT NULL,
    l_shipdate      DATE NOT NULL,
    l_commitdate    DATE NOT NULL,
    l_receiptdate   DATE NOT NULL,
    l_shipinstruct  CHAR(25) NOT NULL,
    l_shipmode      CHAR(10) NOT NULL,
    l_comment       VARCHAR(44) NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);
"@

$schemaSQL | psql -U $DB_USER -d $DB_NAME

Write-Host ""
Write-Host "Step 6: Loading data..." -ForegroundColor Yellow

$tables = @("nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem")
foreach ($table in $tables) {
    Write-Host "Loading $table..."
    $filePath = (Resolve-Path "$DATA_DIR\raw\$table.tbl").Path
    $copyCmd = "\COPY $table FROM '$filePath' DELIMITER '|' CSV;"
    $copyCmd | psql -U $DB_USER -d $DB_NAME
}

Write-Host ""
Write-Host "Step 7: Creating indexes..." -ForegroundColor Yellow

$indexSQL = @"
CREATE INDEX idx_supplier_nationkey ON supplier(s_nationkey);
CREATE INDEX idx_partsupp_partkey ON partsupp(ps_partkey);
CREATE INDEX idx_partsupp_suppkey ON partsupp(ps_suppkey);
CREATE INDEX idx_customer_nationkey ON customer(c_nationkey);
CREATE INDEX idx_orders_custkey ON orders(o_custkey);
CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_partkey ON lineitem(l_partkey);
CREATE INDEX idx_lineitem_suppkey ON lineitem(l_suppkey);
CREATE INDEX idx_lineitem_part_supp ON lineitem(l_partkey, l_suppkey);

VACUUM ANALYZE;
"@

$indexSQL | psql -U $DB_USER -d $DB_NAME

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "TPC-H Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Database: $DB_NAME" -ForegroundColor Cyan
Write-Host "Test with: psql -U $DB_USER -d $DB_NAME -c 'SELECT COUNT(*) FROM lineitem;'" -ForegroundColor Cyan
Write-Host ""