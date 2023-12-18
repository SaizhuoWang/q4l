#!/bin/bash

# Log directory
read -p "Log dir [default: ./logs]: " log_dir
log_dir=${log_dir:-"./logs"}

# Log path
read -p "Please input the path of the directory to store the logs: " log_path
mkdir -p "$log_dir/$log_path"

# PostgreSQL
read -p "PostgreSQL Data Directory [default: $log_dir/$log_path/pg_data]: " pg_data
pg_data=${pg_data:-"$log_dir/$log_path/pg_data"}
mkdir -p "$pg_data"

read -p "PostgreSQL IP Address [default: localhost]: " pg_ip
pg_ip=${pg_ip:-"localhost"}

read -p "PostgreSQL Port [default: 5432]: " pg_port
pg_port=${pg_port:-"5432"}

echo "Initializing PostgreSQL at $pg_data"
initdb -D "$pg_data"

echo "host    all             all             192.168.0.0/16          trust" | cat >> "$pg_data/pg_hba.conf"

echo "Starting PostgreSQL at address postgresql://$pg_ip:$pg_port"
pg_ctl -D "$pg_data" -l "$log_dir/$log_path/pg.log" -o "-p $pg_port -h $pg_ip" start

echo 'Creating a PostgreSQL database named mlflow'
createdb -h "$pg_ip" -p "$pg_port" mlflow

# MongoDB
read -p "MongoDB Data Directory [default: $log_dir/$log_path/mongo_data]: " mongo_data
mongo_data=${mongo_data:-"$log_dir/$log_path/mongo_data"}
mkdir -p "$mongo_data"

read -p "MongoDB IP Address [default: localhost]: " mongo_ip
mongo_ip=${mongo_ip:-"localhost"}

read -p "MongoDB Port [default: 27017]: " mongo_port
mongo_port=${mongo_port:-"27017"}

echo "Starting MongoDB at address mongodb://$mongo_ip:$mongo_port"
mongod --dbpath "$mongo_data" --fork --logpath "$log_dir/$log_path/mongo.log" --bind_ip "$mongo_ip" --port "$mongo_port"

# mlflow
cd "$log_dir/$log_path"

read -p "mlflow IP Address [default: localhost]: " mlflow_ip
mlflow_ip=${mlflow_ip:-"localhost"}

read -p "mlflow Port [default: 5000]: " mlflow_port
mlflow_port=${mlflow_port:-"5000"}

mkdir -p "$log_dir/$log_path/mlflow_artifacts"
echo "Starting mlflow server at address http://$mlflow_ip:$mlflow_port"
mlflow server \
    --host "$mlflow_ip" \
    --port "$mlflow_port" \
    --workers 10 \
    --backend-store-uri "postgresql://$(whoami)@$mlflow_ip:$pg_port/mlflow" \
    --default-artifact-root ./mlflow_artifacts
