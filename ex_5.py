import os

import torch
import torch.nn as nn
import numpy as np

# Device configuration
from gcommand_dataset import GCommandLoader, is_audio_file
from my_model import MyModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_height = 161
input_width = 101
num_of_classes = 30

classes = {0: "bed", 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight", 6: "five", 7: "four", 8: "go", 9: "happy",
           10: "house", 11: "left", 12: "marvin", 13: "nine", 14: "no", 15: "off", 16: "on", 17: "one", 18: "right",
           19: "seven", 20: "sheila", 21: "six", 22: "stop", 23: "three", 24: "tree", 25: "two", 26: "up", 27: "wow",
           28: "yes", 29: "zero"}


def train(model, train_loader, num_of_epochs, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        audio, label = data
        audio = audio.to(device)
        label = label.to(device)
        output = model(audio)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += label.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum().item()
    print('Epoch [{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(num_of_epochs + 1, loss.item(),
                                                               (correct / total) * 100))
    return model


def validate(model, val_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for audio, label in val_loader:
            audio = audio.to(device)
            label = label.to(device)
            output = model(audio)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        acc = 100 * correct / (len(val_loader) * 100)
        print('Test Accuracy of the model on the validation: {:.2f} %'.format((correct / total) * 100))
    return model


def create_test_list():
    test_names = []
    dir = os.path.expanduser('./gcommands/test')
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk((d))):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    test_names.append(fname)
    return test_names


def sort_test_result_to_file(test_names, classes_names):
    predict_list = []
    for i in range(len(test_names)):
        try:
            index = test_names.index(str(i) + ".wav")
            predict_list.append(test_names[index] + "," + classes_names[index])
        except:
            try:
                index = test_names.index(str(i) + ".WAV")
                predict_list.append(test_names[index] + "," + classes_names[index])
            except:
                continue
    return predict_list


def prediction_to_file(predict_list):
    with open("test_y", "w") as file_test:
        for y in predict_list:
            file_test.write(y)
            file_test.write("\n")


def prediction_check(model, test_loader):
    test_names = create_test_list()
    model.eval()
    test_list, classes_list = [], []
    i = 0
    with open("test_y_check", "w") as file_test:
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            # label = label.to(device)
            output = model(data)
            predict = output.cpu().data.max(1, keepdim=True)[1]
            name_of_class = classes[predict.item()]
            test_list.append(test_names[i])
            classes_list.append(name_of_class)
            file_test.write(test_names[i] + "," + name_of_class + "\n")
            i += 1
        return sort_test_result_to_file(test_list, classes_list)


def prediction(model, test_loader):
    test_names = create_test_list()
    model.eval()
    test_list, classes_list = [], []
    i = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        # label = label.to(device)
        output = model(data)
        predict = output.cpu().data.max(1, keepdim=True)[1]
        name_of_class = classes[predict.item()]
        test_list.append(test_names[i])
        classes_list.append(name_of_class)

        i += 1
    return sort_test_result_to_file(test_list, classes_list)


def train_and_test_model(model, train_loader, val_loader, num_of_epochs=25, learning_rate=0.0009):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(num_of_epochs):
        model = train(model, train_loader, i, criterion, optimizer)
    torch.save(model.state_dict(), "model.pt")
    model = validate(model, val_loader)
    return model


def load_data(batch_size):
    train_dataset = GCommandLoader('./gcommands/train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_dataset = GCommandLoader('./gcommands/valid')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, pin_memory=True)

    test_dataset = GCommandLoader('./gcommands/test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


# def train_and_test_model_by_lr(train_loader, val_loader, num_of_epochs=25):
#     criterion = nn.CrossEntropyLoss()
#     x = np.linspace(0.0002, 0.01, num=99)
#
#     max_acc = 0
#     max_lr = 0
#     for learning_rate in x:
#         model = MyModel().to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         for i in range(num_of_epochs):
#             train(train_loader, i, criterion, optimizer)
#         acc = validate(val_loader)
#         if acc > max_acc:
#             print("acc now is {:.2f} % and lr now is {} ".format(acc, learning_rate))
#             max_acc = acc
#             max_lr = learning_rate
#             print("max_acc is {:.2f} % and max_lr is {} ".format(max_acc, max_lr))


def main():
    batch_size = 100
    train_loader, val_loader, test_loader = load_data(batch_size)
    model = MyModel().to(device)
    model = train_and_test_model(model, train_loader, val_loader)
    # train_and_test_model_by_lr(train_loader, val_loader)
    predict_list = prediction_check(model, test_loader)
    prediction_to_file(predict_list)


if __name__ == '__main__':
    main()

# todo: params, report
