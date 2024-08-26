# Описание

Используйте скрипт train.py для обучения.

Есть также класс embedder_train.processing.data_splitter.StratifiedDataSplitter для деления датафрейма на train-val-test выборки в нужном соотношении, напрямую не использую при обучении, но решил все равно добавить.

# Гиперпараметры

Гиперпараметры, влияющие на потребляемую память и на скорость обучения:
1. gradient_checkpointing
3. per_device_train_batch_size желательно побольше, >=128
4. per_device_eval_batch_size
5. distance_metric, cosine или dot
6. torch_compile
7. dataloader_pin_memory
8. max_seq_length

влияющие на качество обучения:
1. предобработка датасета
1. max_steps
2. warmup_ratio
3. learning_rate
7. loss_func, margin
4. per_device_train_batch_size
6. distance_metric
9. metric_for_best_model
5. max_seq_length

Тут подробно описаны все гиперпараметры, влияюшие на обучение, для библиотеки transformers 
https://huggingface.co/docs/transformers/perf_train_gpu_one