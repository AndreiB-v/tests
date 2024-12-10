from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def import1():
    test_data = [1, 2, 3]
    return test_data


def import2():
    test_data = [3, 2, 1]
    return test_data


def import3():
    test_data = [2, 1, 3]
    return test_data


def data_processing(**kwargs):
    ti = kwargs['ti']
    d1 = ti.xcom_pull(task_ids='import1')
    d2 = ti.xcom_pull(task_ids='import2')
    d3 = ti.xcom_pull(task_ids='import3')

    all_done = d1 + d2 + d3
    return all_done

def save_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='data_processing_task')
    with open('/home/bub/Рабочий стол/data.txt', 'w') as f:
        f.write(str(data))


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(weeks=1),
}

with DAG(
        'test_imports',
        default_args=default_args,
        description='Test imports DAG',
        schedule_interval=timedelta(days=1),
        catchup=False,
) as dag:
    import1_task = PythonOperator(
        task_id='import1',
        python_callable=import1,
        do_xcom_push=True,
        # do_xcom_push используется для передачи информации из одного таска в другой
    )

    import2_task = PythonOperator(
        task_id='import2',
        python_callable=import2,
        do_xcom_push=True,
    )

    import3_task = PythonOperator(
        task_id='import3',
        python_callable=import3,
        do_xcom_push=True,
    )

    data_processing_task = PythonOperator(
        task_id='data_processing_task',
        python_callable=data_processing,
        do_xcom_push=True,
    )

    save_data_task = PythonOperator(
        task_id='save_data_task',
        python_callable=save_data
        # do_xcom_push не устанавливаем, так как функция ничего не возвращает
    )

    [import1_task, import2_task, import3_task] >> data_processing_task >> save_data_task
