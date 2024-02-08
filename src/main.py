import sys

from PyQt5.QtWidgets import QApplication
from src.gui import MainMenu
import os

if __name__ == '__main__':
    os.environ[
        "QT_QPA_PLATFORM_PLUGIN_PATH"] = ('/home/ams/anaconda3/envs/efficient-net/lib/python3.8/site-packages/cv2/qt'
                                          '/plugins/platforms')

    app = QApplication(sys.argv)
    main_window = MainMenu()
    main_window.show()
    sys.exit(app.exec_())

    # #args = get_args()
    #
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #
    # class_names = [name for name in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, name))]
    # num_classes = len(class_names)
    # print(args.weights)
    #
    # model = ModelBuilder(
    #     name=args.model_name,
    #     pretrained=args.pretrained,
    #     fine_tune=args.fine_tune,
    #     num_classes=num_classes,
    #     model_name=args.model_name,
    #     weights=args.weights,
    # )
    #
    # if args.mode == 'train':
    #     from ai_operations.models import Trainer
    #     from ai_operations.tools import DataPreparation
    #
    #     data_prep = DataPreparation(
    #         root_dir=args.input,
    #         pretrained=args.pretrained,
    #     )
    #
    #     trainer = Trainer(
    #         data_preparation=data_prep,
    #         model=model,
    #         learning_rate=args.learning_rate,
    #         epochs=args.epochs,
    #
    #     )
    #
    #     trainer.run()
    #
    # elif args.mode == 'metrics':
    #
    #     from ai_operations.inference import Metrics
    #
    #     metrics = Metrics(
    #         model_builder=model,
    #         input_dir=args.input,
    #         output_dir=args.output
    #     )
    #
    #     metrics.obtain_metrics()

# TODO: poner en builder model la opcion de inferencia entonces asi no lo tengoq ue poner cada vez que lo hago y que devuelva los outputs, tambien camiar el nomnre.
# TODO: modifica readme para explicar wandb