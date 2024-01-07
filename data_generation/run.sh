CONFIG_NAME='config001'
YAML_PATH1=./config/${CONFIG_NAME}.yaml
BASE_SAVEDIR=/path/to/savedir/

python mk_data.py --worker_id 0 \
                  --yaml_path ${YAML_PATH1} \
                  --base_save_dir ${BASE_SAVEDIR}
python create_json.py --data_path ${BASE_SAVEDIR}/${CONFIG_NAME}
