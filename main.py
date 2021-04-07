from src import data, model, features


def main():
    data.fetch_data()
    features.split_data()
    features.transform_data(train=True)
    model.train_data()
    features.transform_data(test=True)
    model.evaluate_model()


if __name__ == '__main__':
    main()
