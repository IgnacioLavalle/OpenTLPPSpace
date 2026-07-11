#!/bin/bash

set -e

run_experiment() {
    local nombre="$1"
    shift

    local total=0

    echo
    echo "===== $nombre ====="

    for i in {1..5}
    do
        echo "--- Corrida $i ---"

        SECONDS=0

        python3 "$@"

        local tiempo=$SECONDS
        total=$((total + tiempo))

        echo "Tiempo corrida $i: ${tiempo} s"
        echo
    done

    promedio=$(awk "BEGIN {printf \"%.2f\", $total/5}")

    echo "--------------------------------------"
    echo "Tiempo total: ${total} s"
    echo "Tiempo promedio: ${promedio} s"
    echo "======================================"
    echo
}

echo "######## Grafo pesado - Dataset completo ########"

run_experiment "DDNE" DDNE_fc_oos.py
run_experiment "ELD" eld_fc.py
run_experiment "D2V" d2v_fc.py
run_experiment "TMF" TMF_fc.py
run_experiment "LIST" LIST_FC.py

echo "######## Grafo no pesado - Dataset completo ########"

run_experiment "TMF no pesado" tmf_uw_fc.py
run_experiment "LIST no pesado" list_uw_fc.py

echo "######## Grafo pesado - Dataset recortado ########"

run_experiment "GCNGAN recortado" gcngan_fc_oos.py
run_experiment "DDNE recortado" DDNE_fc_oos.py --data_name Recortado677
run_experiment "ELD recortado" eld_fc.py --data_name Recortado677
run_experiment "D2V recortado" d2v_fc.py --data_name Recortado677
run_experiment "TMF recortado" TMF_fc.py --data_name Recortado677
run_experiment "LIST recortado" LIST_FC.py --data_name Recortado677

echo "######## Todos los experimentos finalizaron ########"
