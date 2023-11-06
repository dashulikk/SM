import numpy as np
import matplotlib.pyplot as plt

from matplotlib import image


def moore_penrose_method(matrix, sigma0, eps=1e-5):
    # Step 1 (sigma0 init by user)

    matrix = np.array(matrix, dtype=float)
    e = np.eye(matrix.shape[0])
    sigma_k = sigma0

    # Step 2
    plus_matrix = matrix.T @ np.linalg.inv(matrix @ matrix.T + sigma0 * e)

    while True:
        # Step 3
        sigma_k = sigma_k / 2

        previous = plus_matrix

        # Step 4
        plus_matrix = matrix.T @ np.linalg.inv(matrix @ matrix.T + sigma_k * e)

        # Step 5
        if np.linalg.norm(plus_matrix - previous) < eps:
            return plus_matrix


def greville_method(matrix):
    matrix = np.array(matrix, dtype=float)

    # Get first row
    a = matrix[0:1]

    if np.count_nonzero(a[0]) == 0:
        result = np.zeros_like(a.T)
    else:
        result = a.T / a @ a.T

    # Greville formula
    for i in range(1, matrix.shape[0]):
        z_a = np.eye(result.shape[0]) - result @ matrix[:i]
        r_a = result @ result.T
        a = matrix[i:i + 1]

        dot_product = (a @ z_a) @ a.T

        if np.count_nonzero(dot_product) == 0:
            part_a = (r_a @ a.T) / (1 + (a @ r_a) @ a.T)
        else:
            part_a = (z_a @ a.T) / dot_product

        result = np.concatenate((result - part_a @ (a @ result), part_a), axis=1)

    return result


def pseudoinverse_matrix_check(x_plus, x):
    result = True

    result = result and ((x @ x_plus) @ x).all() == x.all()
    result = result and ((x_plus @ x) @ x_plus).all() == x_plus.all()
    result = result and np.allclose(x @ x_plus, (x @ x_plus).T)
    result = result and np.allclose(x_plus @ x, (x_plus @ x).T)

    return result


def main():
    x_image, y_image = image.imread('x1.bmp'), image.imread('y7.bmp')

    moore_penrose_matrix = moore_penrose_method(x_image, 1)
    greville_matrix = greville_method(x_image)

    moore_penrose_status = 'Ok' if pseudoinverse_matrix_check(moore_penrose_matrix, x_image) else 'Bad'
    print(f'Status Moore-Penrose method: {moore_penrose_status}')

    greville_status = 'Ok' if pseudoinverse_matrix_check(greville_matrix, x_image) else 'Bad'
    print(f'Status Greville method: {greville_status}')

    moore_penrose_operator = y_image @ moore_penrose_matrix
    greville_operator = y_image @ greville_matrix

    # Show result
    fig = plt.figure()

    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('X')
    plt.imshow(x_image, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Y')
    plt.imshow(y_image, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('Moore-Penrose')
    plt.imshow(moore_penrose_operator @ x_image, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('Greville')
    plt.imshow(greville_operator @ x_image, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()