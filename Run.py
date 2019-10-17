import digits
import mnisttest
import matplotlib.pyplot as plt

if __name__ == '__main__':
    acc = 100
    acc_max = 1
    acc_min = 100
    values = []
    # while acc > 85:
    for i in range(30):
        digits.run('training-images.txt', 'training-labels.txt', 'validation-images.txt')
        acc = mnisttest.run('results.txt', 'validation-labels.txt')
        # print(acc)
        values.append(acc)
        if (acc > acc_max):
            acc_max = acc
            print('max: ' + str(acc_max))
        if (acc < acc_min):
            acc_min = acc
            print('min: ' + str(acc_min))

    plt.plot(values)
    plt.plot([85 for i in range(30)])
    plt.title('Alpha change 0.1 and later 0.2')
    plt.show()