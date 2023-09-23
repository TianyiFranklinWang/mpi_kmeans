#include <omp.h>
#include <mpi.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

const int NUM_THREADS{4};
const int RAND_SEED{42};
const int EARLY_STOPPING_EPOCH{25};

using namespace std;

template<typename T>
class Array2D {
    T **data_ptr;
    unsigned m_rows;
    unsigned m_cols;

    T **create2DArray(unsigned nrows, unsigned ncols, const T &val = T()) {
        T **ptr = nullptr;
        T *pool = nullptr;
        try {
            ptr = new T *[nrows];  // allocate pointers (can throw here)
            pool = new T[nrows * ncols]{val};  // allocate pool (can throw here)

            // now point the row pointers to the appropriate positions in
            // the memory pool
            for (unsigned i = 0; i < nrows; ++i, pool += ncols)
                ptr[i] = pool;

            // Done.
            return ptr;
        }
        catch (std::bad_alloc &ex) {
            delete[] ptr; // either this is nullptr or it was allocated
            throw ex;  // memory allocation error
        }
    }

public:
    typedef T value_type;

    T **data() {
        return data_ptr;
    }

    unsigned get_rows() const {
        return m_rows;
    }

    unsigned get_cols() const {
        return m_cols;
    }

    Array2D() : data_ptr(nullptr), m_rows(0), m_cols(0) {}

    Array2D(unsigned rows, unsigned cols, const T &val = T()) {
        if (rows == 0)
            throw std::invalid_argument("number of rows is 0");
        if (cols == 0)
            throw std::invalid_argument("number of columns is 0");
        data_ptr = create2DArray(rows, cols, val);
        m_rows = rows;
        m_cols = cols;
    }

    ~Array2D() {
        if (data_ptr) {
            delete[] data_ptr[0];  // remove the pool
            delete[] data_ptr;     // remove the pointers
        }
    }

    Array2D(const Array2D &rhs) : m_rows(rhs.m_rows), m_cols(rhs.m_cols) {
        data_ptr = create2DArray(m_rows, m_cols);
        std::copy(&rhs.data_ptr[0][0], &rhs.data_ptr[m_rows - 1][m_cols], &data_ptr[0][0]);
    }

    Array2D(Array2D &&rhs) noexcept {
        data_ptr = rhs.data_ptr;
        m_rows = rhs.m_rows;
        m_cols = rhs.m_cols;
        rhs.data_ptr = nullptr;
    }

    Array2D &operator=(Array2D &&rhs) noexcept {
        if (&rhs != this) {
            swap(rhs, *this);
            rhs.data_ptr = nullptr;
        }
        return *this;
    }

    void swap(Array2D &left, Array2D &right) {
        std::swap(left.data_ptr, right.data_ptr);
        std::swap(left.m_cols, right.m_cols);
        std::swap(left.m_rows, right.m_rows);
    }

    Array2D &operator=(const Array2D &rhs) {
        if (&rhs != this) {
            Array2D temp(rhs);
            swap(*this, temp);
        }
        return *this;
    }

    T *operator[](unsigned row) {
        return data_ptr[row];
    }

    const T *operator[](unsigned row) const {
        return data_ptr[row];
    }

    void create(unsigned rows, unsigned cols, const T &val = T()) {
        *this = Array2D(rows, cols, val);
    }
};

void arg_parser(const int &, char **, string &, string &, unsigned int &, unsigned int &, unsigned int &);

void read_input(const string &, unsigned int &, unsigned int &, Array2D<long double> &);

void split_observations(const unsigned int &, const int &, int *, int *);

void scatter_observations(const int &, const int *, const int *, unsigned int &, const Array2D<long double> &,
                          Array2D<long double> &);

template<typename T>
T random(T, T);

template<typename T, typename P>
T weighted_random(P *, int);

void
kmeanspp_initialize(const int &, const unsigned int &, int *, int *, const unsigned int &, const Array2D<long double> &,
                    const Array2D<long double> &, Array2D<long double> &);

void
compute_one_iter(const int &, const unsigned int &, const Array2D<long double> &, Array2D<long double> &,
                 long double &);

void write_output(const string &, const long double &, const Array2D<long double> &);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    cout.precision(numeric_limits<long double>::max_digits10);

    srand(RAND_SEED + world_rank);

    string input_file_name, output_file_name;
    unsigned int k, n_features, max_iter;
    if (world_rank == 0) {
        arg_parser(argc, argv, input_file_name, output_file_name, k, n_features, max_iter);
    }

    if (world_rank == 0) {
        cout << "--------- Initializing Process ---------" << endl;
        for (int i = 0; i < world_size; ++i) {
            bool ready{false};
            if (i == 0) {
                ready = true;
            } else {
                MPI_Recv(&ready, 1, MPI_CXX_BOOL, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (ready) {
                cout << "    - Process ready! [" << i + 1 << "/" << world_size << "]" << endl;
            }
        }
    } else {
        bool ready{true};
        MPI_Send(&ready, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD);
    }

    Array2D<long double> observations;
    int *counts{nullptr}, *displacements{nullptr};
    if (world_rank == 0) {
        cout << endl << "--------- Reading Input ---------" << endl;
        unsigned int num_observations{0};
        read_input(input_file_name, n_features, num_observations, observations);
        cout << "    - Number of observations: " << num_observations << endl;

        cout << endl << "--------- Scattering observations ---------" << endl;
        counts = new int[world_size];
        displacements = new int[world_size];
        split_observations(num_observations, world_size, counts, displacements);
        for (int i = 0; i < world_size; ++i) {
            counts[i] *= n_features;
            displacements[i] *= n_features;
        }
        for (int i = 0; i < world_size; ++i) {
            cout << "    - Rank " << i << "  counts: " << counts[i] << "  displacement: " << displacements[i] << endl;
        }
    }
    Array2D<long double> my_observations;
    scatter_observations(world_rank, counts, displacements, n_features, observations, my_observations);

    MPI_Bcast(&k, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    Array2D<long double> centroids(k, n_features);
    if (world_rank == 0) {
        cout << endl << "--------- Initializing with k-means++ ---------" << endl;

        for (int i = 0; i < world_size; ++i) {
            counts[i] /= n_features;
            displacements[i] /= n_features;
        }
    }
    kmeanspp_initialize(world_rank, k, counts, displacements, n_features, observations, my_observations, centroids);
    if (world_rank == 0) {
        cout << "    - Initial centroids:" << endl;
        for (unsigned int i = 0; i < k; ++i) {
            cout << "\t";
            for (unsigned int j = 0; j < n_features; ++j) {
                if (j)
                    cout << "  ";
                cout << centroids[i][j];
            }
            cout << endl;
        }

        cout << endl << "--------- Updating centroids with K-means ---------" << endl;
    }

    MPI_Bcast(&max_iter, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    bool early_stopping{false};
    int early_stopping_counter{0};
    long double old_loss{0};
    long double loss{0};
    for (unsigned int i = 0; i < max_iter; ++i) {
        if (early_stopping) {
            if (world_rank == 0) {
                cout << endl << "\t*** Early stopping! ***" << endl;
            }
            break;
        }

        loss = 0;
        compute_one_iter(world_rank, n_features, my_observations, centroids, loss);

        if (world_rank == 0) {
            loss /= observations.get_rows();
            if (i == 0) {
                old_loss = loss;
            } else {
                if (fabs(old_loss - loss) < numeric_limits<double>::epsilon()) {
                    early_stopping_counter += 1;
                } else {
                    early_stopping_counter = 0;
                    old_loss = loss;
                }
            }
            if (early_stopping_counter == EARLY_STOPPING_EPOCH) {
                early_stopping = true;
            }

            cout << "    - Epoch " << i + 1 << "/" << max_iter << ": \t" << "Loss: " << loss << endl;
        }
        MPI_Bcast(&early_stopping, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        cout << endl << "--------- Saving output ---------" << endl;

        cout << "    - Computed centroids:" << endl;
        for (unsigned int i = 0; i < k; ++i) {
            cout << "\t";
            for (unsigned int j = 0; j < n_features; ++j) {
                if (j)
                    cout << "  ";
                cout << centroids[i][j];
            }
            cout << endl;
        }
        write_output(output_file_name, loss, centroids);

        cout << endl << "Complete!" << endl;
    }

    MPI_Finalize();
}

void arg_parser(const int &argc, char **argv, string &input_file_name, string &output_file_name, unsigned int &k,
                unsigned int &n_features, unsigned int &max_iter) {
    if (argc < 2) {
        cerr << "Missing arguments. See 'mpi_omp_kmeans -h/--help' for more info." << endl;
        exit(1);
    }

    if (string(argv[1]) == string("-h") || string(argv[1]) == string("--help")) {
        cout << "Usage: mpirun -n NUM_PROCESS " << argv[0]
             << " INPUT_FILE --k K_CLUSTERS --n_features NUM_FEATURES --out OUTPUT_FILE" << endl
             << "Parallel run K-means clustering algorithm with NUM_PROCESS process(es) on INPUT_FILE,"
             << " which has NUM_FEATURES features, to create K_CLUSTERS clusters and save result to OUTPUT_FILE."
             << endl << endl
             << "Mandatory arguments to long options are mandatory for short options too." << endl
             << "      --k    set number of clusters for algorithm." << endl
             << "      --n_features    tell how many properties are there inside INPUT_FILE." << endl
             << "      --out    set output file." << endl
             << "      --help    display this help and exit." << endl << endl
             << "Code with love by Tianyi Wang at 2022." << endl
             << "Released under MIT license." << endl;
        MPI_Finalize();
        exit(0);
    } else {
        if (argc < 10) {
            cerr << "Missing arguments. See 'mpi_omp_kmeans -h/--help' for more info." << endl;
            exit(1);
        }
        input_file_name = argv[1];
        int flag{0};
        for (int i = 2; i < argc; i += 2) {
            if (string(argv[i]) == "--k") {
                k = stoi(argv[i + 1]);
                flag++;
            }
            if (string(argv[i]) == "--n_features") {
                n_features = stoi(argv[i + 1]);
                flag++;
            }
            if (string(argv[i]) == "--out") {
                output_file_name = argv[i + 1];
                flag++;
            }
            if (string(argv[i]) == "--max-iter") {
                max_iter = stoi(argv[i + 1]);
                flag++;
            }
        }
        if (flag != 4) {
            cerr << "Missing arguments. See 'mpi_omp_kmeans -h/--help' for more info." << endl;
            exit(1);
        }
    }
}


void read_input(const string &file_name, unsigned int &n_features, unsigned int &num_observations,
                Array2D<long double> &observations) {
    ifstream input_file(file_name);
    if (!input_file.is_open()) {
        cerr << "Error open file" << endl;
        exit(1);
    }

    string buffer;
    while (getline(input_file, buffer)) {
        ++num_observations;
    }

    input_file.clear();
    input_file.seekg(0);
    Array2D<long double> tmp(num_observations, n_features);
    for (unsigned int i = 0; i < num_observations; ++i) {
        for (unsigned int j = 0; j < n_features; ++j) {
            input_file >> buffer;
            tmp[i][j] = stold(buffer);
        }
    }
    observations = tmp;

    input_file.close();
}

void split_observations(const unsigned int &num_observations, const int &world_size, int *counts, int *displacements) {
    int fraction_for_none_root = floor(num_observations / world_size);
    int fraction_for_root = num_observations - (world_size - 1) * fraction_for_none_root;

    for (int i = 0; i < world_size; ++i) {
        if (i == 0) {
            counts[i] = fraction_for_root;
            displacements[i] = 0;
        } else {
            counts[i] = fraction_for_none_root;
            if (i == 1) {
                displacements[i] = displacements[i - 1] + fraction_for_root;
            } else {
                displacements[i] = displacements[i - 1] + fraction_for_none_root;
            }
        }
    }
}

void scatter_observations(const int &world_rank, const int *counts, const int *displacements, unsigned int &n_features,
                          const Array2D<long double> &observations, Array2D<long double> &my_observations) {
    int my_count{0};
    MPI_Scatter(counts, 1, MPI_INT, &my_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_features, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    my_observations.create(my_count / n_features, n_features);
    if (world_rank == 0) {
        MPI_Scatterv(&observations[0][0], counts, displacements, MPI_LONG_DOUBLE, &my_observations[0][0], my_count,
                     MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_LONG_DOUBLE, &my_observations[0][0], my_count, MPI_LONG_DOUBLE,
                     0, MPI_COMM_WORLD);
    }
}

template<typename T>
T random(T range_from, T range_to) {
    random_device rand_dev;
    mt19937 generator(rand_dev());
    uniform_int_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

template<typename T, typename P>
T weighted_random(P *weights, int count) {
    random_device rand_dev;
    mt19937 generator(rand_dev());
    discrete_distribution<T> distr(weights, weights + count);
    return distr(generator);
}

void kmeanspp_initialize(const int &world_rank, const unsigned int &k, int *counts, int *displacements,
                         const unsigned int &n_features, const Array2D<long double> &observations,
                         const Array2D<long double> &my_observations, Array2D<long double> &centroids) {
    if (world_rank == 0) {
        auto first_centroid_index = random<unsigned int>(0, observations.get_rows());
        for (unsigned int i = 0; i < n_features; ++i) {
            centroids[0][i] = observations[first_centroid_index][i];
        }
    }
    MPI_Bcast(&centroids[0][0], n_features, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    long double *distances = nullptr;
    if (world_rank == 0) {
        distances = new long double[observations.get_rows()];
    }
    for (unsigned int i = 1; i < k; ++i) {
        auto my_distances = new long double[my_observations.get_rows()];
        for (unsigned int j = 0; j < my_observations.get_rows(); ++j) {
            my_distances[j] = numeric_limits<long double>::infinity();
            for (unsigned int m = 0; m < i; ++m) {
                long double distance{0};
                for (unsigned int n = 0; n < n_features; ++n) {
                    distance += pow((centroids[m][n] - my_observations[j][n]), 2);
                }
                if (distance < my_distances[j]) {
                    my_distances[j] = distance;
                }
            }
        }
        MPI_Gatherv(my_distances, my_observations.get_rows(), MPI_LONG_DOUBLE, distances, counts, displacements,
                    MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
        if (world_rank == 0) {
            auto centroid_index = weighted_random<unsigned int, long double>(distances, observations.get_rows());
            for (unsigned int j = 0; j < n_features; ++j) {
                centroids[i][j] = observations[centroid_index][j];
            }
        }
        MPI_Bcast(&centroids[i][0], n_features, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void
compute_one_iter(const int &world_rank, const unsigned int &n_features, const Array2D<long double> &my_observations,
                 Array2D<long double> &centroids, long double &loss) {
    auto my_cluster_table = new unsigned int[my_observations.get_rows()];
    auto my_distances = new long double[my_observations.get_rows()];
    for (unsigned int i = 0; i < my_observations.get_rows(); ++i) {
        my_distances[i] = numeric_limits<long double>::infinity();
        for (unsigned int j = 0; j < centroids.get_rows(); ++j) {
            long double distance{0};
            for (unsigned int m = 0; m < n_features; ++m) {
                distance += pow((my_observations[i][m] - centroids[j][m]), 2);
            }
            if (distance < my_distances[i]) {
                my_distances[i] = distance;
                my_cluster_table[i] = j;
            }
        }
    }

    Array2D<long double> my_new_centroids(centroids.get_rows(), n_features);
    auto *my_observations_per_centroid = new unsigned int[centroids.get_rows()]{int()};
    long double my_loss = 0;
    for (unsigned int i = 0; i < my_observations.get_rows(); ++i) {
        my_loss += my_distances[i];
        my_observations_per_centroid[my_cluster_table[i]] += 1;
        for (unsigned j = 0; j < n_features; ++j) {
            my_new_centroids[my_cluster_table[i]][j] += my_observations[i][j];
        }
    }

    Array2D<long double> new_centroids(centroids.get_rows(), n_features);
    auto *observations_per_centroid = new unsigned int[centroids.get_rows()]{int()};
    MPI_Allreduce(&my_new_centroids[0][0], &new_centroids[0][0],
                  my_new_centroids.get_rows() * my_new_centroids.get_cols(), MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(my_observations_per_centroid, observations_per_centroid, centroids.get_rows(), MPI_UNSIGNED, MPI_SUM,
                  MPI_COMM_WORLD);
    for (unsigned int i = 0; i < new_centroids.get_rows(); ++i) {
        for (unsigned j = 0; j < n_features; ++j) {
            new_centroids[i][j] /= observations_per_centroid[i];
        }
    }
    centroids = new_centroids;

    MPI_Reduce(&my_loss, &loss, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void write_output(const string &output_file_name, const long double &loss, const Array2D<long double> &centroids) {
    ofstream output_file(output_file_name);
    if (!output_file.is_open()) {
        cerr << "Error open file" << endl;
        exit(1);
    }

    output_file.precision(numeric_limits<long double>::max_digits10);
    output_file << "Loss: " << loss << endl << endl;
    output_file << "Centroids: " << endl;
    for (unsigned int i = 0; i < centroids.get_rows(); ++i) {
        for (unsigned int j = 0; j < centroids.get_cols(); ++j) {
            if (j)
                output_file << "  ";
            output_file << centroids[i][j];
        }
        output_file << endl;
    }
}