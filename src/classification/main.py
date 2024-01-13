from args import get_args
import os
from ai_operations.models import ModelBuilder
import logging

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    class_names = [name for name in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, name))]
    num_classes = len(class_names)
    print(args.weights)

    model = ModelBuilder(
        name=args.model_name,
        pretrained=args.pretrained,
        fine_tune=args.fine_tune,
        num_classes=num_classes,
        model_name=args.model_name,
        weights=args.weights,
    )

    if args.mode == 'train':
        from ai_operations.models import Trainer
        from ai_operations.tools import DataPreparation

        data_prep = DataPreparation(
            root_dir=args.input,
            pretrained=args.pretrained,
        )

        trainer = Trainer(
            data_preparation=data_prep,
            model=model,
            learning_rate=args.learning_rate,
            epochs=args.epochs,

        )

        trainer.run()

    if args.mode == 'metrics':

        from ai_operations.inference import Metrics

        metrics = Metrics(
            model_builder=model,
            input_dir=args.input,
            output_dir=args.output
        )

        metrics.obtain_metrics()







