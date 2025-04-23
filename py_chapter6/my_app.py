#code6.13
#python3 ./py_chapter6/my_app.pyで実行すること
#myenv_ttslearnという仮想環境に入ること
#初回のみpip install hydra-coreを実行すること
import hydra
from omegaconf import DictConfig, OmegaConf

#config_pathは相対パスのみで，このファイルから見た位置．
#ヴァージョン指定
@hydra.main(config_path="../conf_chapter6", config_name="config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
   print(OmegaConf.to_yaml(cfg))
   
if __name__ == "__main__":
   my_app()

