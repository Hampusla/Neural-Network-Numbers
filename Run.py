import digits
import mnisttest
import matplotlib.pyplot as plt
from numpy import diff

if __name__ == '__main__':
    acc = 100
    acc_max = 1
    acc_min = 100
    values = []
    # plt.ylim(140, 200)
    # while acc > 85:
    for i in range(30):
        # plt.plot(digits.run('training-images.txt', 'training-labels.txt', 'validation-images.txt'))
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

    average = sum(values)/len(values)
    print(average)
    plt.plot(values)
    plt.plot([85 for i in range(30)])
    plt.title('145')
    plt.show()

    # Standard average 88,46  88,326666666667
    # 140      average 88,46333333
    # 145      average 88,4933333333335 88,540000000002
    # 170      average 87,96666666665