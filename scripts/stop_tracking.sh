#!/bin/bash

# Initialize an environment for our experiments, including PostgreSQL, MongoDB, and mlflow
source activate mlflow

# Log directory
read -p "Log dir [default: /path/to/logs]: " log_dir
log_dir=${log_dir:-"/path/to/logs"}

# Log path
read -p "Please input the path of the directory to store the logs: " log_path

# PostgreSQL
read -p "PostgreSQL IP Address [default: localhost]: " pg_ip
pg_ip=${pg_ip:-"localhost"}

read -p "PostgreSQL Port [default: 5432]: " pg_port
pg_port=${pg_port:-"5432"}

echo "Stopping PostgreSQL at $log_dir/$log_path/pg_data"
pg_ctl -D "$log_dir/$log_path/pg_data" -l "$log_dir/$log_path/pg.log" -o "-p $pg_port -h $pg_ip" stop

# MongoDB
read -p "MongoDB IP Address [default: localhost]: " mongo_ip
mongo_ip=${mongo_ip:-"localhost"}

read -p "MongoDB Port [default: 27017]: " mongo_port
mongo_port=${mongo_port:-"27017"}

echo "Stopping MongoDB at address mongodb://$mongo_ip:$mongo_port"
mongod --dbpath "$log_dir/$log_path/mongo_data" --shutdown --logpath "$log_dir/$log_path/mongo.log"
