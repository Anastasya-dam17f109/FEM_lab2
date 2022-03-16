import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import os
import os.path

a = 0.25
# набор узлов и получающихся КЭ
nodes = {}
mesh_elems = {}
# форма области задана жестко - это прямоугльник, строящийся по двум точкам coord_min и coord_max
coord_min = np.array([0, 0], float)
coord_max = np.array([2*a, 2*a], float)

# служебная инфа, диспользуется непосредственно в решении
mesh_B_dict = {}
mesh_S_dict = {}
mesh_list_dict = {}
teta_grad = {}
f_func = []
q = {}

# в качестве КЭ используется равнобедренный прямоугольный треугольник, длинна стороны которго = h
h = float(input("Введите размер конечного элемента: "))

# параметры  области
teta = 0.8*10**3
G = 1.74
F_0 = 0



# триангуляция заданной области

def triangulation():
    # создание узлов
    coord_buf = np.zeros(2, dtype=float)
    coord_up = np.zeros(2, dtype=int)
    coord_down = np.zeros(2, dtype=int)
    x_n = int((coord_max[0] - coord_min[0]) // h) + 1
    if (coord_max[0] - coord_min[0]) / h > x_n:
        x_n += 1
    y_n = int((coord_max[1] - coord_min[1]) // h) + 1
    if (coord_max[1] - coord_min[1]) / h > x_n:
        y_n += 1
    counter = 0
    coord_buf[1] = coord_min[1]
    for i in range(x_n):
        coord_buf[1] = coord_max[1]
        for j in range(y_n):
            nodes.update({counter: coord_buf.copy()})
            counter += 1
            if j != y_n - 1:
                coord_buf[1] -= h
            else:
                coord_buf[1] = coord_min[1]
        if i != x_n - 1:
            coord_buf[0] += h
        else:
            coord_buf[0] = coord_max[0]
    counter_mesh = 0
    # создание конечных элементов
    for i in range(x_n - 1):
        for k in range(y_n - 1):
            for l in range(2):
                coord_up[l] = (i + l) * y_n + k
                coord_down[l] = (i + l) * y_n + k + 1
            mesh_elems.update({counter_mesh: np.array([coord_up[0], coord_up[1], coord_down[0]])})
            mesh_elems.update({counter_mesh + 1: np.array([coord_up[1], coord_down[0], coord_down[1]])})
            counter_mesh += 2


# функция , осуществляющая проверку того, что узлы выбранного КЭ лежат на выбранной границе
# непосредственное осуществление проверки в соответствии с заданными признаками

def on_sigma_nodes_list(treshold, idx_el, idx_coord):
    node_list = mesh_elems.get(idx_el)
    curve = []
    buf = 0
    for i in range(3):
        if nodes.get(node_list[i])[idx_coord] == treshold:
            curve.append(node_list[i])
        else:
            buf = node_list[i]
    if len(curve) == 2:
        curve.append(buf)
        return curve
    else:
        return []


# функция , осуществляющая проверку того, что узлы выбранного КЭ лежат на выбранной границе -
# в данной функции задается набор признаков для проверки

def on_sigma_nodes( idx_el):

    # список узлов конечного элемента, лежащих на верхней и нижней границе
    buf_curve = on_sigma_nodes_list(coord_max[0], idx_el, 0)
    if len(buf_curve) != 0:
        return buf_curve
    else:
        return on_sigma_nodes_list(coord_max[1], idx_el, 1)


# вычисление барицентрических координат  узлов элемента

def baricentric(node_list):
    v_matrix = np.ones(9, dtype=float).reshape(3, 3)
    for i in range(3):
        v_matrix[i, 1:] = nodes.get(node_list[i])
    return np.linalg.inv(v_matrix.T)


# вычисление площади конечного элемента

def calc_S(v1, v2, v3):
    p1 = np.array(nodes[v1])
    p2 = np.array(nodes[v2]) - p1
    p3 = np.array(nodes[v3]) - p1
    v_res = np.cross(p2, p3)
    return np.sqrt(np.dot(v_res, v_res)) * 0.5


# вычисление длины стороны конечного элемента

def calc_L(v1, v2):
    v_res = np.array(nodes[v1]) - np.array(nodes[v2])
    return np.sqrt(np.dot(v_res, v_res))

# предикат что узлы принадлежат внешней поверхности

def is_on_sigma_nodes(j_gl):
    if nodes.get(j_gl)[0] == coord_max[0]:
        return True
    else:
        if nodes.get(j_gl)[1] == coord_max[1]:
            return True
        else:
            return False


# сборка глобальной матрицы

def create_global_system():

    for i in range(len(mesh_elems)):
        node_list = mesh_elems.get(i).copy()
        S = calc_S(node_list[0], node_list[1], node_list[2])
        mesh_S_dict.update({i: S})
        node_list = mesh_elems.get(i).copy()
        B = baricentric(node_list.copy())
        mesh_B_dict.update({i: B})
        mesh_list_dict.update({i: node_list})
        f = 2*G*teta*S*np.array([1.0,1.0,1.0])/3
        for j in range(3):
            f_gl_std[node_list[j]] += f[j]

    # учет гу 1го рода
    for i in range(len(mesh_elems)):
        if len(on_sigma_nodes(i)) != 0:
            f = np.array([F_0, F_0])
            node_list = on_sigma_nodes(i)
            for j in range(2):
                i_gl = node_list[j]
                gl_matr_std[i_gl, i_gl] = 1
                f_gl_std[i_gl] = f[j]

    for i in range(len(mesh_elems)):
        B = mesh_B_dict.get(i).copy().T
        B_t = mesh_B_dict.get(i)[:, 1:]
        B = B[1:, :]
        G_matr = mesh_S_dict.get(i) * np.dot(B_t, B)
        for j in range(3):
            i_gl = mesh_list_dict.get(i)[j]
            if not is_on_sigma_nodes(i_gl):
                for k in range(3):
                    j_gl = mesh_list_dict.get(i)[k]
                    if not is_on_sigma_nodes(j_gl):
                        gl_matr_std[i_gl, j_gl] += G_matr[j, k]
                    else:
                        f_gl_std[i_gl] -= G_matr[j, k] * F_0

# вычисление момента кручения

def calc_M_value():
    J_buf = 0
    for i in range(len(mesh_elems)):
        node_list = mesh_elems.get(i).copy()
        J_buf += mesh_S_dict.get(i)*2*(f_func[node_list[0]]+f_func[node_list[1]]+f_func[node_list[2]])/3
# проверка по сегерлинду - не сходятся формулы на teta*G
    print(J_buf)

# вывод результата в файл формата mv2 - для визуаизации

def print_in_mv2():
    with open("./result_mke.txt", 'w') as file: 
        file.write(str(len(nodes)) + ' 3 1 F   \n')
        for i in range(len(nodes)):
            str_buf1 = ''

            for j in range(2):
                str_buf1 += str(nodes.get(i)[j]) + ' '

            str_buf1 += '0 '
            file.write(str(i + 1) + ' ' + str_buf1 + str(f_func[i])  + '\n')
        file.write(str(len(mesh_elems)) + ' 3 3 BC_id mat_id mat_id_Out\n')
        for i in range(len(mesh_elems)):
            str_buf1 = ''
            for j in range(3):
                str_buf1 += str(mesh_list_dict.get(i)[j] + 1) + ' '
            file.write(str(i + 1) + ' ' + str_buf1 + '0 1 0\n')

    file.close()
    if os.path.exists("result_mke.mv2"):
        os.remove("result_mke.mv2")
    os.rename("result_mke.txt", "result_mke.mv2")


# решение поставленной задачи

triangulation()
f_gl_std = np.zeros(len(nodes))
gl_matr_std = np.zeros(len(nodes) * len(nodes)).reshape(len(nodes), len(nodes))
create_global_system()
f_func = np.linalg.solve(gl_matr_std, f_gl_std)
calc_M_value()
print_in_mv2()



