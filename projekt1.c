#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <omp.h>
#include <mpi.h>

#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

#define ROOT 0
int process_id;

typedef struct {
    GLbyte r;
    GLbyte g;
    GLbyte b;
} pixel;

GLuint texture;

#define WIDTH 512
#define HEIGHT 512
#define HALF_WIDTH WIDTH/2
#define HALF_HEIGHT HEIGHT/2
#define N_PIXELS WIDTH*HEIGHT
pixel image_shared[N_PIXELS];
pixel image_local[N_PIXELS];
int generate_directly;

#define GOLDEN_RATIO 1.61803398875
int n_iters;
double zoom;
double offset_x, offset_y;
int julia;
double julia_x, julia_y;

// 0 threads means max possible
void generatePixels(int n_threads, int offset, int length) {
    int pixel_i, i;
    int row, col;
    double scaled_x, scaled_y;
    double x, y;
    double temp_x;
    double t;
    double time, time_seconds;

    int* count;
    if (n_threads == 0) n_threads = omp_get_max_threads();
    count = (int*)calloc(n_threads, sizeof(int));

    if (generate_directly) time = omp_get_wtime();
    else time = MPI_Wtime();

    // 1 task load is 1 pixel, scheduling dynamic since time to calculate may vary
#pragma omp parallel for private(row,col,scaled_x,scaled_y,i,x,y,temp_x,t) num_threads(n_threads) schedule(dynamic,1)
    for (pixel_i = offset; pixel_i < offset + length; pixel_i++) {
        row = pixel_i / HEIGHT;
        col = pixel_i % HEIGHT;

        x = (col - HALF_WIDTH) / (zoom * HALF_WIDTH) + offset_x;
        y = (row - HALF_HEIGHT) / (zoom * HALF_HEIGHT) + offset_y;
        if (julia) {
            scaled_x = julia_x;
            scaled_y = julia_y;
        }
        else {
            // mandelbrot
            scaled_x = x;
            scaled_y = y;
        }
        i = 0;
        while (i < n_iters && (x*x + y*y) < 4) {
            temp_x = x*x - y*y + scaled_x;
            y = 2*x*y + scaled_y;
            x = temp_x;
            i++;
        }

        if (i == n_iters) {
            // did not escape
            image_local[pixel_i - offset].r = (GLbyte) 0;
            image_local[pixel_i - offset].g = (GLbyte) 0;
            image_local[pixel_i - offset].b = (GLbyte) 0;
        }
        else {
            // color depends on number of iterations it took to escape
            t = 1.0 * i / n_iters;
            // modified Bernstein polynomials
            image_local[pixel_i - offset].r = (GLbyte) (15 * (1 - t) * (1 - t) * t * t * 255);
            image_local[pixel_i - offset].g = (GLbyte) (9 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            image_local[pixel_i - offset].b = (GLbyte) (9 * (1 - t) * t * t * t * 255);
        }
        if (generate_directly) {
            // save directy to root array, only when 1 process
            image_shared[pixel_i].r = image_local[pixel_i - offset].r;
            image_shared[pixel_i].g = image_local[pixel_i - offset].g;
            image_shared[pixel_i].b = image_local[pixel_i - offset].b;
        }

        count[omp_get_thread_num()]++;
    }

    if (generate_directly) {
        time = omp_get_wtime() - time;
        time_seconds = time;
    } else {
        time = MPI_Wtime() - time;
        MPI_Reduce(&time, &time_seconds, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    }

    // !! MPI does not sync cout
    printf("process responsible for pixels %d to %d\n", offset, offset+length-1);
    for (i = 0; i < n_threads; i++) {
        printf("\tthread %d: %d pixels\n", i, count[i]);
    }
    if (process_id == ROOT) {
        printf("%.8f seconds\n\n", time_seconds);
        //fprintf(stderr, "%.8f\n", time_seconds);
    }
    fflush(stdout);
}

// Initialize OpenGL state
void init() {
    // Texture setup
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Other
    glClearColor(0, 0, 0, 0);
    gluOrtho2D(-1, 1, -1, 1);
    glLoadIdentity();
    glColor3f(1, 1, 1);
}

// Generate and display the image.
void display() {
    // Copy image to texture memory
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, image_shared);
    // Clear screen buffer
    glClear(GL_COLOR_BUFFER_BIT);
    // Render a quad
    glBegin(GL_QUADS);
        glTexCoord2f(1,0); glVertex2f(1,-1);
        glTexCoord2f(1,1); glVertex2f(1,1);
        glTexCoord2f(0,1); glVertex2f(-1,1);
        glTexCoord2f(0,0); glVertex2f(-1,-1);
    glEnd();
    // Display result
    glFlush();
    glutPostRedisplay();
    glutSwapBuffers();
}

void keypress(unsigned char key, int x, int y) {
    switch (key) {
        case '+':
            n_iters += 32;
            break;
        case '-':
            n_iters -= 32;
            break;
        case 'i':
        case 'I':
            zoom *= 2;
            break;
        case 'o':
        case 'O':
            zoom /= 2;
            break;
        case 'w':
        case 'W':
            offset_y += 0.1 / zoom;
            break;
        case 'a':
        case 'A':
            offset_x -= 0.1 / zoom;
            break;
        case 's':
        case 'S':
            offset_y -= 0.1 / zoom;
            break;
        case 'd':
        case 'D':
            offset_x += 0.1 / zoom;
            break;
        default: return;
    }
    generatePixels(0, 0, N_PIXELS);
    display();
}

// Main entry function
int main(int argc, char **argv) {
    int n_processes, chunk_size, n_threads = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    chunk_size = N_PIXELS / n_processes;

    if (argc >= 2) n_threads = atoi(argv[1]);

    julia = 0;
    if (argc == 4) {
        julia = 1;
        julia_x = atof(argv[2]);
        julia_y = atof(argv[3]);
    }

    n_iters = 128;
    zoom = 0.5;
    offset_x = offset_y = 0.0;
    if (0) {
        // TESTING
        julia = 1;
        julia_x = GOLDEN_RATIO - 2;
        julia_y = GOLDEN_RATIO - 1;
        n_iters = 1024;
        zoom = 68719476736.0;
        offset_x = 0.37060449865530254;
        offset_y = -0.35494532230368331;
    }

    if (process_id == ROOT) {
        if (julia) printf("drawing Julia set c = %+f %+fi\n", julia_x, julia_y);
        else printf("drawing Mandelbrot set\n");
        fflush(stdout);
    }
    generate_directly = 0;
    generatePixels(n_threads, process_id*chunk_size, chunk_size);

    if (process_id == ROOT) {
        MPI_Gather(image_local, chunk_size * sizeof(pixel), MPI_BYTE, \
                   image_shared, chunk_size * sizeof(pixel), MPI_BYTE, \
                   ROOT, MPI_COMM_WORLD);
    }
    else {
        MPI_Gather(&image_local, chunk_size * sizeof(pixel), MPI_BYTE, \
                   &image_shared, chunk_size * sizeof(pixel), MPI_BYTE, \
                   ROOT, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    if (process_id == ROOT) {
        // eventual keyboard changes later will manage 1 process only
        generate_directly = 1;
        // Init GLUT
        glutInit(&argc, argv);
        glutInitWindowSize(WIDTH, HEIGHT);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
        glutCreateWindow("Fractal");
        // Set up OpenGL state
        init();
        // Run the control loop
        glutDisplayFunc(display);
        glutKeyboardFunc(keypress);
        glutMainLoop();
    }

    return EXIT_SUCCESS;
}
