/*
CSCI 480
Assignment 3 Raytracer
Name: <Ju Young (Amy) Lee>
*/

#include <pic.h>
#include <windows.h>
#include <stdlib.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <stdio.h>
#include <string>
#include <math.h>
using namespace std;

#define MAX_TRIANGLES 2000
#define MAX_SPHERES 2000
#define MAX_LIGHTS 2000

char *filename=0;

//different display modes
#define MODE_DISPLAY 1
#define MODE_JPEG 2
int mode = MODE_DISPLAY;

//you may want to make these smaller for debugging purposes
#define WIDTH 640
#define HEIGHT 480

//the field of view of the camera
#define fov 60.0
#define PI 3.141592
double aspect = (double)WIDTH/(double)HEIGHT;

//enum for 2D barycentric coordinates
typedef enum { XY, YZ, XZ } PLANETYPE;
PLANETYPE rt_planetype;

// structs // 
struct Vertex
{
	double position[3];
	double color_diffuse[3];
	double color_specular[3];
	double normal[3];
	double shininess;
};

struct Point {
	double x;
	double y;
	double z;
} ;

typedef struct _Triangle
{
	struct Vertex v[3];
} Triangle;

typedef struct _Sphere
{
	double position[3];
	double color_diffuse[3];
	double color_specular[3];
	double shininess;
	double radius;
} Sphere;

typedef struct _Light
{
	double position[3];
	double color[3];
} Light;


// other variables //
unsigned char buffer[HEIGHT][WIDTH][3];
double color [HEIGHT][WIDTH][3];
double normal [HEIGHT][WIDTH][3];
double diffuse [HEIGHT][WIDTH][3];
double specular [HEIGHT][WIDTH][3];
double shine [HEIGHT][WIDTH];
double ray[3];

Point lightP, interP; //interP = is the coordinate of most updated intersection of view ray
//lightP = coordinate of the intersection of shadow ray
Point sumColor;

Triangle triangles[MAX_TRIANGLES];
Sphere spheres[MAX_SPHERES];
Light lights[MAX_LIGHTS];
double ambient_light[3];

//initialize//
int num_triangles=0;
int num_spheres=0;
int num_lights=0;

// functions //

void plot_pixel_display(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel_jpeg(int x,int y,unsigned char r,unsigned char g,unsigned char b);
void plot_pixel(int x,int y,unsigned char r,unsigned char g,unsigned char b);

//rayTracer figures out the intersects and the shadows
void rayTracer(int x, int y, double x0, double y0, double z0, double* viewRay);
void sphereIntersect (int x , int y, double x0, double y0, double z0, double *viewRay, double &dist);
void triangleIntersect(int x , int y, double x0, double y0, double z0, double *viewRay, double &dist);

//shadowrayTracer implements the illumination equation
void shadowrayTracer(int x, int y, double x0, double y0, double z0, double* viewRay);
void shadowSphereIntersect(int x, int y, double x0, double y0, double z0,double *viewRay, double &dist, int i, double lightMag);
void shadowTriangleIntersect(int x, int y, double x0, double y0, double z0, double *viewRay,  double &dist,int i, double lightMag);

//arithmetic functions
double calculateArea(Vertex V1, Vertex V2, Vertex V3,PLANETYPE rt_planetype);
void normalizeVector(double &X, double &Y, double &Z, double &norm);



void draw_scene()
{
	unsigned int x,y;
	double radFov = fov*(double)2*PI/(double)360; //get the radiance of fov angle
	double norm;
	//simple output
	for(x=0; x<WIDTH; x++)
	{
		glPointSize(2.0);  
		glBegin(GL_POINTS);

		for(y = 0; y < HEIGHT;y++)
		{
			//draw arrays//
			//x = -a*tan(fov/2) to +a*tan(fov/2) //
			ray[0] = -aspect*tan(radFov/(double)2) + ((double)x / WIDTH) * 2 * aspect * tan(radFov/(double)2);
			//y = -tan(fov/2) to +tan(fov/2) //
			ray[1] = -tan(radFov/(double)2) + ((double)y/HEIGHT)*2*tan(radFov/(double)2);
			//z = -1 since we assume camera is at 0,0,0 facing -z direction
			ray[2] = -1;

			//normalize rays//
			normalizeVector(ray[0],ray[1],ray[2],norm);
			
			rayTracer((double)x,(double)y,0,0,0,ray);

			//draws the pixel with the colors determined from rayTracer
			plot_pixel(x,y, (color[y][x][0])*255,(color[y][x][1])*255,(color[y][x][2])*255);
		}
		glEnd();
		glFlush();
	}

	printf("Done!\n"); fflush(stdout);
}



void rayTracer(int x, int y, double x0, double y0, double z0, double* viewRay)
{

	double distance = -1;

	sphereIntersect(x, y, x0, y0, z0, viewRay, distance);
	triangleIntersect(x, y, x0, y0, z0, viewRay, distance);

	//if distance == -1, there was no interesction form sphere or triangle
	if(distance == -1)
	{
		//set color to white because it doesn't intersect with anything
		color[y][x][0] = 1.0;
		color[y][x][1] = 1.0;
		color[y][x][2] = 1.0;
	}
	else {// if there was an intersection, fire a shadow ray if blocked, shadow, if not use phong
		shadowrayTracer(x, y, interP.x, interP.y, interP.z, viewRay);
	}

}

void sphereIntersect (int x , int y, double x0, double y0, double z0, double *viewRay, double &dist)
{
	double b, c, delta, t0, t1; // variables from ray-sphere intersection slide 
	//delta = the quantity under the sqrt ( b^2 -4c )
	Point sCenter; // x, y, z of the sphere's center
	double radius;

	//looking for interesction along the spheres
	for (int i = 0; i<num_spheres; i++)
	{
		sCenter.x = spheres[i].position[0];
		sCenter.y = spheres[i].position[1];
		sCenter.z = spheres[i].position[2];

		radius = spheres[i].radius;

		b = (double)2 * (viewRay[0]*(x0 - sCenter.x) + viewRay[1]*(y0 - sCenter.y) + viewRay[2]*(z0 - sCenter.z));
		c = pow((x0 - sCenter.x),2) + pow((y0 - sCenter.y),2) + pow((z0 - sCenter.z),2) - pow(radius,2) ;
		delta = pow(b,2) - ((double)4*c);
		
		/*
		DETERMINIG THE INTERSECTION BY THE DELTA VALUE
		If B2-4AC<0, the ray and sphere do not intersect
		If B2-4AC=0, ray grazes sphere
		If B2-4AC>0, the smallest positive t corresponds to the intersection 
		*/

		if (delta<0) //if you can't compute the b^2-4ac then they don't intersect
		{
			//color it white for background//
			color[y][x][0] = 1.0; 
			color[y][x][1] = 1.0;
			color[y][x][2] = 1.0;
		}
		else //delta is greater than 0, so it's intersected
		{
			t0 = (-b + (sqrt(delta)))/(double)2;
			t1 = (-b - (sqrt(delta)))/(double)2;

			if ((t0<0) && (t1>0)) //if t0 is a negative number then take t1
				t0 = t1;
			else if ((t0>0) && (t1<0)) // if t1 is negative, take t0
				t0 = t0;
			else if (t0>0 && t1>0) //both positive then take the min
				t0 = min(t0,t1);
			
			//only if t0 is positive you can compute or else it should be negative
			if((t0> 0) && (dist == -1 || t0 < dist)) 
			{
				
					dist = t0;
					interP.x = x0 + viewRay[0]*t0;
					interP.y = y0 + viewRay[1]*t0;
					interP.z = z0 + viewRay[2]*t0;

					for (int coordinate= 0;coordinate<3;coordinate++)
					{
						diffuse[y][x][coordinate] = spheres[i].color_diffuse[coordinate];
						specular[y][x][coordinate] = spheres[i].color_specular[coordinate];
					}

					shine[y][x] = spheres[i].shininess;

					normal[y][x][0] = (interP.x - sCenter.x)/radius;
					normal[y][x][1] = (interP.y - sCenter.y)/radius;
					normal[y][x][2] = (interP.z - sCenter.z)/radius;

			}
		
					
			
		}
	}
	

}


void triangleIntersect(int x , int y, double x0, double y0, double z0, double *viewRay, double &dist)
{
	Vertex A, B, C, interPoint;

	Point p1, p2, Ntriangle;//Ntriangle = normal of the triangle

	double norm, D, T; //norm = magnitude of normal to normalize the normals 
	// D = -ax - by -cz;
	// T = determines intersection point in plane, if T <= 0, the intersection is behind ray origin
	double areaDenom, alphaNum, betaNum, gammaNum;
	double alpha, beta, gamma;



	for (int i = 0; i < num_triangles; i++)
	{
		//get the vertex information of triangles
		A = triangles[i].v[0];
		B = triangles[i].v[1];
		C = triangles[i].v[2]; 

		//calculate the normal (B-A) x (C-A) //
		// p1 = B-A and p2 = C-A //
		p1.x = B.position[0] - A.position[0];
		p1.y = B.position[1] - A.position[1];
		p1.z = B.position[2] - A.position[2];

		p2.x = C.position[0] - A.position[0];
		p2.y = C.position[1] - A.position[1];
		p2.z = C.position[2] - A.position[2];

		// N = p1 x p2 = |[i j k; p1.x p1.y p1.z; p2.x p2.y p2.z]|
		Ntriangle.x = (p1.y * p2.z) - (p1.z * p2.y);
		Ntriangle.y = (p1.z * p2.x) - (p1.x * p2.z); 
		Ntriangle.z = (p1.x * p2.y) - (p1.y * p2.x);

		normalizeVector(Ntriangle.x,Ntriangle.y,Ntriangle.z,norm);

		D  = (-Ntriangle.x * A.position[0]) - (Ntriangle.y * A.position[1]) - (Ntriangle.z * A.position[2]);

		//if Ntriangle dot myray = 0, no intersection 
		if (((Ntriangle.x * viewRay[0]) + (Ntriangle.y * viewRay[1]) + (Ntriangle.z * viewRay[2])) != 0)
		{
			
			//to figure out the intersection ///
			T = -((Ntriangle.x * x0) + (Ntriangle.y * y0) + (Ntriangle.z * z0) + D)/(Ntriangle.x * viewRay[0] + Ntriangle.y * viewRay[1] + Ntriangle.z * viewRay[2]);
			
			interPoint.position[0] = x0 + viewRay[0]*T;
			interPoint.position[1] = y0 + viewRay[1]*T;
			interPoint.position[2] = z0 + viewRay[2]*T;


			//projecting triangles to 2d planes to make it easier
			if (Ntriangle.x != 0)
			{
				rt_planetype = YZ;
			}
			else if(Ntriangle.y !=0)
			{
				rt_planetype = XZ;
			}
			else //Nz !=0
			{
				rt_planetype = XY;
			}

			
			// alphaNum = area(C C1 C2) = 1/2[(C1-C) x (C2-C)]
			// betaNum = area(C0 C C2) =  1/2[(C-C0) x (C2-C0)]
			// gammaNum = area (C0 C1 C) = 1/2[(C1-C0) x (C-C0)]

			alphaNum = calculateArea(interPoint,B,C,rt_planetype);
			betaNum = calculateArea(A,interPoint,C,rt_planetype);
			gammaNum = calculateArea(A, B, interPoint,rt_planetype);
			areaDenom = calculateArea(A,B,C,rt_planetype);

			if(rt_planetype == YZ)
				alphaNum = double(-1)*alphaNum;


			//barycentric coordinates in 2D

			alpha = alphaNum/areaDenom;
			beta = betaNum/areaDenom;
			gamma = gammaNum/areaDenom;


			// the point is inside the triangle if alpha + gamma + beta =1 && 0<= alpha, beta, gamma <=1
			// and T > 0
			double barycSum = alpha + beta + gamma;
			if(( barycSum <= 1.05) && (barycSum >= 0.95)) //cannot exactly be 1 so give 0.05 error
			{
				if((alpha>=0) && (alpha<=1) && (beta>=0) && (beta<=1) && (gamma>=0) && (gamma<=1) && (T>0))
				{
					//find the closest intersection (where T<dist) and calculate the color of pixel
					//initially distance == -1 (if there was no intersection in the sphere)
					if (dist == -1 || (T<dist)) //or the new T is closer than the previous intersection
					{
						//new distance becasuse we got a closer intersection
						dist = T;
						//update the interesction point
						interP.x = interPoint.position[0];
						interP.y = interPoint.position[1];
						interP.z = interPoint.position[2];


						for (int xyz =0; xyz<3;xyz++)
						{
							normal[y][x][xyz] = alpha*A.normal[xyz] + beta*B.normal[xyz] + gamma*C.normal[xyz];
							diffuse[y][x][xyz] = alpha*A.color_diffuse[xyz] + beta*B.color_diffuse[xyz] + gamma*C.color_diffuse[xyz];
							specular[y][x][xyz] = alpha*A.color_specular[xyz] + beta*B.color_specular[xyz] + gamma*C.color_specular[xyz];
						}
						shine[y][x] = alpha*A.shininess + beta*B.shininess + gamma*C.shininess;

					}

				}
			}


		}

	}


}




void shadowSphereIntersect(int x, int y, double x0, double y0, double z0, double *viewRay, double &dist, int i, double lightMag)
{
	int counter =0;
	Point sphereP, N, R, newRay;
	double LdotN, RdotV;
	double radius, b, c, delta, t0, t1;

	//if shadow ray intersects to spheres, color should be black//
	for (int j = 0;j < num_spheres;j++)
	{
		sphereP.x = spheres[j].position[0];
		sphereP.y = spheres[j].position[1];
		sphereP.z = spheres[j].position[2];

		radius = spheres[j].radius;

		b = (double)2 * (lightP.x*(x0 - sphereP.x) + lightP.y * (y0 - sphereP.y) + lightP.z * (z0 - sphereP.z));
		c = pow((x0 - sphereP.x),2) + pow((y0 - sphereP.y),2) + pow((z0 - sphereP.z),2) - pow(radius,2) ;
		delta = pow(b,2) - (double)4*c;

			t0 = (-b + (sqrt(delta)))/(double)2;
			t1 = (-b - (sqrt(delta)))/(double)2;

			//only care about what the new dist is //
			if ( (dist == -1) || (dist> t0) || (dist>t1) )
			{
				if(t0 > 0 && t1> 0)
				{
					dist = min(t0,t1);
				}
				else if (t0>0)
				{
					dist = t0;
				}
				else if (t1>0)
				{
					dist = t1;
				}

			}

			//GLOBAL ILLUMINATION//
			//I = lightColor * (kd * (L dot N) + ks * (R dot V) ^ sh) 
			//ANGLE OF RELFECTION//
			//R = 2(L dot N) * N - L 

			N.x = normal[y][x][0];
			N.y = normal[y][x][1];
			N.z = normal[y][x][2];

			double norm;
			normalizeVector(N.x,N.y,N.z,norm);
			

			//L dot N = Light.x * Normal.x  + Light.y * Normal.y + Light.z * Normal.z
			LdotN = (lightP.x * N.x) + (lightP.y * N.y) + (lightP.z * N.z);
			R.x = ((double)2 * LdotN * N.x) - lightP.x;
			R.y = ((double)2 * LdotN * N.y) - lightP.y;
			R.z = ((double)2 * LdotN * N.y) - lightP.z;

			double Rnorm;
			normalizeVector(R.x,R.y,R.z,Rnorm);
		
			RdotV = -R.x*viewRay[0] - R.y*viewRay[1] - R.z*viewRay[2];

			//clamping 
			if (LdotN < 0)
				LdotN = 0;
			if (RdotV < 0)
				RdotV = 0;

			if( (delta > 0) && (dist != -1))
			{
				//ray equation
				newRay.x = x0 + (lightP.x*dist);
				newRay.y = y0 + (lightP.y*dist);
				newRay.z = z0 + (lightP.z*dist);

				double magnitude = sqrt(pow(newRay.x,2) + pow(newRay.y,2) + pow(newRay.z,2));
				//if this ray is shorter than the shadow ray that was launched, it's in the shadow
				if(magnitude <= lightMag)
				{
					//shadow
					color[y][x][0] = 0.0;
					color[y][x][1] = 0.0;
					color[y][x][2] = 0.0;

				}

			}
			else //if delta isn't greater 0 then, no intersection add the lights and colors
			{
				if(i == counter)// only adds once for one light 
					// i is a counter for the num_lights
				{
					sumColor.x += lights[i].color[0] * (diffuse[y][x][0] * LdotN + specular[y][x][0]*pow(RdotV,shine[y][x]));
					sumColor.y += lights[i].color[1] * (diffuse[y][x][1] * LdotN + specular[y][x][1]*pow(RdotV,shine[y][x]));
					sumColor.z += lights[i].color[2] * (diffuse[y][x][2] * LdotN + specular[y][x][2]*pow(RdotV,shine[y][x]));
					
					sumColor.x += ambient_light[0];
					sumColor.y += ambient_light[1];
					sumColor.z += ambient_light[2];
					counter ++;
				}
				


			}
				
		
			//clamping the total color
			if(sumColor.x >1)
				sumColor.x = (double)1.0;
			if(sumColor.y > 1)
				sumColor.y = (double)1.0;
			if(sumColor.z >1)
				sumColor.z = (double)1.0;

			color[y][x][0] = sumColor.x;
			color[y][x][1] = sumColor.y;
			color[y][x][2] = sumColor.z;



	}
}

void shadowTriangleIntersect(int x, int y, double x0, double y0, double z0, double *viewRay,  double &dist, int i, double lightMag)
{
	//same as before but now interesting with light//

	int counter =0;
	Vertex A, B, C, interPoint;

	Point p1, p2, Ntriangle , interP, newRay, N, R;//Ntriangle = normal of the triangle

	double norm, D, T; //norm = magnitude of normal to normalize the normals 
	double LdotN, RdotV;

	//D = -ax - by -cz;
	// T = determines intersection point in plane, if T <= 0, the intersection is behind ray origin
	double areaDenom, alphaNum, betaNum, gammaNum;
	double alpha, beta, gamma;

	for (int k = 0; k<num_triangles;k++)
	{
		//triangle vertices
		A= triangles[k].v[0];
		B= triangles[k].v[1];
		C= triangles[k].v[2];


		//calculate the normal (B-A) x (C-A) //
		// p1 = B-A and p2 = C-A //
		p1.x = B.position[0] - A.position[0];
		p1.y = B.position[1] - A.position[1];
		p1.z = B.position[2] - A.position[2];

		p2.x = C.position[0] - A.position[0];
		p2.y = C.position[1] - A.position[1];
		p2.z = C.position[2] - A.position[2];

		// N = p1 x p2 = |[i j k; p1.x p1.y p1.z; p2.x p2.y p2.z]|
		Ntriangle.x = (p1.y * p2.z) - (p1.z * p2.y);
		Ntriangle.y = (p1.z * p2.x) - (p1.x * p2.z); 
		Ntriangle.z = (p1.x * p2.y) - (p1.y * p2.x);

		double norm;
		normalizeVector(Ntriangle.x,Ntriangle.y, Ntriangle.z, norm);
		
		D  = (-Ntriangle.x * A.position[0]) - (Ntriangle.y * A.position[1]) - (Ntriangle.z * A.position[2]);

		//if Ntriangle dot lightray = 0, no intersection 
		if((Ntriangle.x*lightP.x + Ntriangle.y*lightP.y + Ntriangle.z*lightP.z) != 0)
		{
			T = -(Ntriangle.x * x0 + Ntriangle.y * y0 + Ntriangle.z * z0 + D)/(Ntriangle.x * lightP.x + Ntriangle.y * lightP.y + Ntriangle.z * lightP.z);

			//projecting triangles to 2d planes to make it easier
			// alphaNum = area(C C1 C2) = 1/2[(C1-C) x (C2-C)]
			// betaNum = area(C0 C C2) =  1/2[(C-C0) x (C2-C0)]
			// gammaNum = area (C0 C1 C) = 1/2[(C1-C0) x (C-C0)]


			interPoint.position[0] = x0 + lightP.x*T;
			interPoint.position[1] = y0 + lightP.y*T;
			interPoint.position[2] = z0 + lightP.z*T;

			if (Ntriangle.x != 0)
			{
				rt_planetype = YZ;
			}
			else if(Ntriangle.y !=0)
			{
				rt_planetype = XZ;
			}
			else //Nz !=0
			{
				rt_planetype = XY;
			}


			alphaNum = calculateArea(interPoint,B,C,rt_planetype);
			betaNum = calculateArea(A,interPoint,C,rt_planetype);
			gammaNum = calculateArea(A, B, interPoint,rt_planetype);
			areaDenom = calculateArea(A,B,C,rt_planetype);

			//barycentric coordinates in 2D

			alpha = alphaNum/areaDenom;
			beta = betaNum/areaDenom;
			gamma = gammaNum/areaDenom;

			double sum = alpha + beta+ gamma;

			// light hits the object //
			if ( (sum <= 1.05) && (sum>=0.95) )
			{
				if ( (alpha<=1) && (alpha >= 0) && (beta>=0) && (beta<=1) && (gamma<=1) && (gamma >= 0) && (T>0) )
				{
					if ((dist > T) || (dist == -1)) 
					{
					dist = T;
					newRay.x = x0 + (lightP.x * dist);
					newRay.y = y0 + (lightP.y * dist);
					newRay.z = z0 + (lightP.z * dist);
					double mag = sqrt( pow(newRay.x,2) + pow(newRay.y,2) + pow(newRay.z,2));

					if(mag <= lightMag)
					{
						//shadow
						for(int cor = 0; cor<3; cor++){
							color[y][x][cor] = 0.0;}

					}
					else
					{
						//I = lightColor * (kd * (L dot N) + ks * (R dot V) ^ sh) 
						//R = 2(L dot N) * N - L

						N.x = normal[y][x][0];
						N.y = normal[y][x][1];
						N.z = normal[y][x][2];

						double normalize = sqrt( pow(N.x,2) + pow(N.y,2) + pow(N.z,2));
						N.x /= normalize;
						N.y /= normalize;
						N.z /= normalize;

						LdotN = N.x * lightP.x + N.y *lightP.y + N.z * lightP.z;
						R.x = 2*LdotN * N.x - lightP.x;
						R.y = 2*LdotN * N.y - lightP.y;
						R.z = 2*LdotN * N.z - lightP.z;

						double Rnorm;
						normalizeVector(R.x,R.y,R.z,Rnorm);
						

						//V is negative of the light ray
						RdotV = -R.x*viewRay[0] - R.y*viewRay[1] - R.z*viewRay[2];

						//clamp
						if(LdotN <0)
							LdotN = 0;
						if (RdotV < 0)
							RdotV = 0;
						if (i == counter)
						{
							sumColor.x += lights[i].color[0] * (diffuse[y][x][0] * LdotN + specular[y][x][0]*pow(RdotV,shine[y][x]));
							sumColor.y += lights[i].color[1] * (diffuse[y][x][1] * LdotN + specular[y][x][1]*pow(RdotV,shine[y][x]));
							sumColor.z += lights[i].color[2] * (diffuse[y][x][2] * LdotN + specular[y][x][2]*pow(RdotV,shine[y][x]));
							
							sumColor.x += ambient_light[0];
							sumColor.y += ambient_light[1];
							sumColor.z += ambient_light[2];
							counter++;
						}
						
					    if(sumColor.x > 1)
							sumColor.x = 1.0;
						if (sumColor.y > 1)
							sumColor.y = 1.0;
						if(sumColor.z > 1)
							sumColor.z = 1.0;
						color[y][x][0] = sumColor.x;
						color[y][x][1] = sumColor.y;
						color[y][x][2] = sumColor.z;


					}

						
						
				}
			}
			else if(dist!=-1)
			{
				for(int cord =0; cord<3;cord++){
					color[y][x][cord] = 0.0;}

				

			}


			}
			

		}

	}

}


void shadowrayTracer(int x, int y, double x0, double y0, double z0, double* viewRay)
{
	//go through the light sources and add the colors all up//
	//initialize//
	sumColor.x = 0;
	sumColor.y = 0;
	sumColor.z = 0;
	//////////////

	for (int i = 0; i <num_lights; i++)
	{

		lightP.x = lights[i].position[0]-x0;
		lightP.y = lights[i].position[1]-y0;
		lightP.z = lights[i].position[2]-z0;

		double lightMag;
		normalizeVector(lightP.x, lightP.y, lightP.z, lightMag);
		
		//set distance to -1 to check intersection
		double distance = -1;
		x0 = x0 + (lightP.x*0.01);
		y0 = y0 + (lightP.y*0.01);
		z0 = z0 + (lightP.z*0.01);

		//sphere interesection//
		shadowSphereIntersect(x,y,x0,y0,z0,viewRay,distance,i, lightMag);
		
		//triangle intersection//
		shadowTriangleIntersect(x,y,x0,y0,z0,viewRay,distance,i, lightMag);


	}

}

//figures out the magnitude and normalizes the given vectors
void normalizeVector(double &X, double &Y, double &Z, double &magnitude)
{
	magnitude = sqrt(pow(X,2) + pow(Y,2) + pow(Z,2));
	X /= magnitude;
	Y /= magnitude;
	Z /= magnitude;
}
//returns area of desired
double calculateArea(Vertex V0, Vertex V1, Vertex V2, PLANETYPE rt_planetype)
{
	double area;
	if(rt_planetype == XY)  // XY plane
	{
		area= (double)0.5*( (V1.position[0]-V0.position[0]) * (V2.position[1]-V0.position[1]) - (V1.position[1]-V0.position[1]) * (V2.position[0] - V0.position[0]) );
	}
	else if(rt_planetype == YZ)
	{
		area = (double)0.5 * ( (V1.position[1]-V0.position[1]) * (V2.position[2]-V0.position[1]) - (V1.position[2]-V0.position[2]) * (V2.position[1] - V0.position[1]) );

	}
	else if(rt_planetype == XZ)
	{
		area = (double)0.5 * ( (V1.position[2]-V0.position[2]) * (V2.position[0]-V0.position[0]) - (V1.position[0]-V0.position[0]) * (V2.position[2] - V0.position[2]) );
	}
	else
	{
		printf("wrong plane, check parameter");
	}
	return area;

}


void plot_pixel_display(int x,int y,unsigned char r,unsigned char g,unsigned char b)
{
	glColor3f(((double)r)/256.f,((double)g)/256.f,((double)b)/256.f);
	glVertex2i(x,y);
}

void plot_pixel_jpeg(int x,int y,unsigned char r,unsigned char g,unsigned char b)
{
	buffer[HEIGHT-y-1][x][0]=r;
	buffer[HEIGHT-y-1][x][1]=g;
	buffer[HEIGHT-y-1][x][2]=b;
}

void plot_pixel(int x,int y,unsigned char r,unsigned char g, unsigned char b)
{
	plot_pixel_display(x,y,r,g,b);
	if(mode == MODE_JPEG)
		plot_pixel_jpeg(x,y,r,g,b);
}

void save_jpg()
{
	Pic *in = NULL;

	in = pic_alloc(640, 480, 3, NULL);
	printf("Saving JPEG file: %s\n", filename);

	memcpy(in->pix,buffer,3*WIDTH*HEIGHT);
	if (jpeg_write(filename, in))
		printf("File saved Successfully\n");
	else
		printf("Error in Saving\n");

	pic_free(in);      

}

void parse_check(char *expected,char *found)
{
	if(stricmp(expected,found))
	{
		char error[100];
		printf("Expected '%s ' found '%s '\n",expected,found);
		printf("Parse error, abnormal abortion\n");
		exit(0);
	}

}

void parse_doubles(FILE*file, char *check, double p[3])
{
	char str[100];
	fscanf(file,"%s",str);
	parse_check(check,str);
	fscanf(file,"%lf %lf %lf",&p[0],&p[1],&p[2]);
	printf("%s %lf %lf %lf\n",check,p[0],p[1],p[2]);
}

void parse_rad(FILE*file,double *r)
{
	char str[100];
	fscanf(file,"%s",str);
	parse_check("rad:",str);
	fscanf(file,"%lf",r);
	printf("rad: %f\n",*r);
}

void parse_shi(FILE*file,double *shi)
{
	char s[100];
	fscanf(file,"%s",s);
	parse_check("shi:",s);
	fscanf(file,"%lf",shi);
	printf("shi: %f\n",*shi);
}

int loadScene(char *argv)
{
	FILE *file = fopen(argv,"r");
	int number_of_objects;
	char type[50];
	int i;
	Triangle t;
	Sphere s;
	Light l;
	fscanf(file,"%i",&number_of_objects);

	printf("number of objects: %i\n",number_of_objects);
	char str[200];

	parse_doubles(file,"amb:",ambient_light);

	for(i=0;i < number_of_objects;i++)
	{
		fscanf(file,"%s\n",type);
		printf("%s\n",type);
		if(stricmp(type,"triangle")==0)
		{

			printf("found triangle\n");
			int j;

			for(j=0;j < 3;j++)
			{
				parse_doubles(file,"pos:",t.v[j].position);
				parse_doubles(file,"nor:",t.v[j].normal);
				parse_doubles(file,"dif:",t.v[j].color_diffuse);
				parse_doubles(file,"spe:",t.v[j].color_specular);
				parse_shi(file,&t.v[j].shininess);
			}

			if(num_triangles == MAX_TRIANGLES)
			{
				printf("too many triangles, you should increase MAX_TRIANGLES!\n");
				exit(0);
			}
			triangles[num_triangles++] = t;
		}
		else if(stricmp(type,"sphere")==0)
		{
			printf("found sphere\n");

			parse_doubles(file,"pos:",s.position);
			parse_rad(file,&s.radius);
			parse_doubles(file,"dif:",s.color_diffuse);
			parse_doubles(file,"spe:",s.color_specular);
			parse_shi(file,&s.shininess);

			if(num_spheres == MAX_SPHERES)
			{
				printf("too many spheres, you should increase MAX_SPHERES!\n");
				exit(0);
			}
			spheres[num_spheres++] = s;
		}
		else if(stricmp(type,"light")==0)
		{
			printf("found light\n");
			parse_doubles(file,"pos:",l.position);
			parse_doubles(file,"col:",l.color);

			if(num_lights == MAX_LIGHTS)
			{
				printf("too many lights, you should increase MAX_LIGHTS!\n");
				exit(0);
			}
			lights[num_lights++] = l;
		}
		else
		{
			printf("unknown type in scene description:\n%s\n",type);
			exit(0);
		}
	}
	return 0;
}

void display()
{

}

void init()
{
	glMatrixMode(GL_PROJECTION);
	glOrtho(0,WIDTH,0,HEIGHT,1,-1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(1,1,1,0);
	glClear(GL_COLOR_BUFFER_BIT);
}

void idle()
{
	//hack to make it only draw once
	static int once=0;
	if(!once)
	{
		draw_scene();
		if(mode == MODE_JPEG)
			save_jpg();
	}
	once=1;
}

int main (int argc, char ** argv)
{
	if (argc<2 || argc > 3)
	{  
		printf ("usage: %s <scenefile> [jpegname]\n", argv[0]);
		exit(0);
	}
	if(argc == 3)
	{
		mode = MODE_JPEG;
		filename = argv[2];
	}
	else if(argc == 2)
		mode = MODE_DISPLAY;


	glutInit(&argc,argv);
	loadScene(argv[1]);

	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
	glutInitWindowPosition(0,0);
	glutInitWindowSize(WIDTH,HEIGHT);
	int window = glutCreateWindow("Ray Tracer");
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	init();
	glutMainLoop();
}

