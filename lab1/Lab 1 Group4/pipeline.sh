#!/bin/bash

# Каталог для скриптов python
script_dir=$1

# Каталог для данных и модели
data_dir=$2

# Название используемого датасета
dataset_name=$3

if  [ -z  "${data_dir}" ]; then
  data_dir="$(pwd)/content"
fi

if  [ -z  "${script_dir}" ]; then
  script_dir="$(pwd)"
fi
if [[ ! -d "$script_dir" && ! -L "$script_dir" ]] ; then
    echo "The specified script directory does not exist."
    echo " The program is over."
    exit 3
fi
if [ !  -r "$script_dir" ] ; then
  echo "The scripts directory cannot be accessed."
  echo "The program is over."
  exit 4
fi

if  [ -z  "${data_type}" ]; then
  dataset_name='moons'
fi

mkdir -m ug+rw -p "$data_dir"
mkdir -m ug+rw -p "$data_dir/test"
mkdir -m ug+rw -p "$data_dir/train"

if [[  ! -r "$data_dir" || ! -w "$data_dir" ]] ; then
  echo "The data directory cannot be accessed."
  echo "The program is over."
  exit 2
fi

# Если venv не установлен:
if [ $(dpkg-query -W -f='${Status}' 'python3-venv' | grep -c 'ok installed') -eq 0 ];
then
  sudo apt-get install python3-venv
fi

venv_dir="$script_dir/env"

if [[ ! -d "$venv_dir" && ! -L "$venv_dir" ]] ; then
  python3 -m venv "$venv_dir"
fi
source "$venv_dir/bin/activate"
pip install -r requirements.txt &> /dev/null

python_scripts=([0]='data_creation.py' [1]='data_preprocessing.py'  [2]='model_preparation.py'  [3]='model_testing.py')

for python_script in "${python_scripts[@]}"
do
  full_path=$script_dir/$python_script
  if [ -s "$full_path" ]; then
    echo "Executing $python_script"
    chmod ugo+x "$full_path"

    if  python3 "$full_path" -d "$data_dir" -t "$dataset_name"
    then
      echo   "$python_script is done"
    else
      echo    "$python_script is not done"
    fi
  else
    echo "$python_script not found"
  fi
done

deactivate