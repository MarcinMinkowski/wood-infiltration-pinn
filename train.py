from trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop(200)
    trainer.save_weights()