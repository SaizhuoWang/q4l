Training deep learning models
================
.. _Pytorch Lightning:  https://lightning.ai/docs/pytorch/stable/


In QuantBench/q4l we use `Pytorch Lightning`_ to train deep learning models. So the training routine follows its convention, where most of the training logic is implemented in a templated class called `LightningModule`. You may refer to :code:`q4l/mode/base.py` to see how it is implemented.


Model Architecture
------------------
We currently support `TemporalModel` that is essentially time-series models, and `SpatiotemporalModel` that incorporates graph neural networks to capture information among stocks. Please refer to API doc for the workflow of these two types of models.


Loss function
------------------
Loss function is implemented as part of the `LightningModule` class. You may refer to :code:`q4l/mode/base.py` to see how it is implemented.


Model inference and generate outputs
------------------
After a model is trained, we need to perform inference on the test set and generate outputs for evaluation (alpha evaluation/backtest/risk analysis/etc.). This is done by the `predict` method of the `LightningModule` class. You may refer to :code:`q4l/mode/base.py` to see how it is implemented.

An example modeling workflow
------------------
Here we provide an example of how to combine the data module and modeling parts to train a model, test its performance, export model inference results and hand these results over to the evaluation module (from qlib) to perform alpha evaluation and backtest. You may also find this script in :code:`examples/benchmark/src/pipeline.py`.

.. code-block:: python

    # Get a logger instance
    logger = get_logger("q4l.task_fn")
    seed_everything(job_config.misc.seed)
    logger.info(f"job_config.misc.device = {job_config.misc.device}")

    # Log the dataset name and load the dataset
    logger.info(f"Loading data ...")
    data = Q4LDataModule(exp_config=exp_config, job_config=job_config)
    logger.info(f"Successfully loaded dataset {data}")
    # logger.info(
    #     f"Memory profile after creating dataset: {display_memory_tree(ProcessNode(psutil.Process()))}"
    # )

    # Create the model and optimizer
    logger.info(f"Creating model ...")
    exp_config.model.model_type
    model: QuantModel = quant_model_factory(exp_config, data=data)
    logger.info(f"Successfully created model {model}")
    # logger.info(
    #     f"Memory profile after creating model: {display_memory_tree(ProcessNode(psutil.Process()))}"
    # )
    if not isinstance(model, NonDLModel):
        model_statistics = summary(
            model,
            depth=3,
        )
        R.set_tags(model_params=model_statistics.total_params)

    # Create the trainer and train the model
    logger.info("Creating trainer")
    # torch.set_float32_matmul_precision("medium")
    strategy = SingleDeviceStrategy(device=job_config.misc.device)
    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(R.artifact_uri, "checkpoints"),
        filename="model_{epoch number}",
        monitor=exp_config.model.basic_info.valid_metric,
        save_last=True,
        save_top_k=1,
        verbose=True,
        mode="max",
    )
    es_callback = EarlyStopping(
        monitor=exp_config.model.basic_info.valid_metric,
        mode="max",
        patience=exp_config.model.basic_info.patience,
        verbose=True,
    )
    tbl = TensorBoardLogger(
        save_dir=job_config.machine.tensorboard_dir,
        name=job_config.name.run_name,
    )
    csvl = CSVLogger(save_dir=R.artifact_uri, name="csv_logs")
    profiler = PyTorchProfiler(
        filename="profile.txt",
        record_module_names=True,
        profile_memory=True,
        group_by_input_shape=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        with_modules=True,
        record_shapes=True,
    )
    trainer = Q4LTrainer(
        default_root_dir=R.artifact_uri,
        strategy=strategy,
        callbacks=[ckpt_callback, es_callback],
        logger=[tbl, csvl],
        enable_progress_bar=job_config.misc.debug,
        # profiler=profiler,
        **exp_config.model.trainer,
    )

    # Train the model
    logger.info("Starts training.")
    if exp_config.model.model_type != "rl":
        trainer.fit(model, datamodule=data)
    else:
        trainer.fit(model)
    logger.info("Training finished")

    # Evaluate the model on the test set and save the checkpoint
    if exp_config.model.model_type != "rl":
        trainer.predict(model=model, datamodule=data)
    else:
        trainer.predict(model=model)

    recorder_config_dict = OmegaConf.to_container(
        exp_config.collector.single, resolve=True
    )

    # Record the performance
    logger.info("Start recording performance")

    recorder = R.get_recorder()
    for record_name, recorder_config in recorder_config_dict.items():
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # to make defining the tasking easier
        logger.info(f"Running recorder {record_name}")
        r = create_instance(
            recorder_config,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": data},
            recorder=recorder,
            recorder_wrapper=R,
        )
        r.generate()

    logger.info("Successfully recorded performance, task finished.")
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time} seconds")
