#adalah sebuah script yang bisa digunakan untuk melakukan training
from tqdm import tqdm

class train():
    """Merupakan kelas yang akan digunakan untuk meproses latihan""" 
    def __init__(self, model, data_loader, loss_fn, optimizer) -> None:
        #memasukkan semua nilai - nilai yang ada 
        self.model = model
        self.data_loader = data_loader 
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def train_loop(self, epoch):
        #adalah trainig loop yang akan dibuat
        acc_list = []
        loss_list = []

        for x, y in enumerate(tqdm(self.data_loader)):
            y = self.model(x)

            loss ="barebones"


    def eval(self, epoch):
        #adalah untuk menandakan bahwa saatnya melakukan evaluasi
        self.model.eval()

        print("yess ini dia yang saya suka")

    