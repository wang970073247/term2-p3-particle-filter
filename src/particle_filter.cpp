/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define NUM_PARTICLES 100
#define LOW_LIMIT 0.001

using namespace std;
default_random_engine rand_num;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = NUM_PARTICLES;
	normal_distribution<double> d_x(x, std[0]);
	normal_distribution<double> d_y(y, std[1]);
	normal_distribution<double> d_theta(theta, std[2]);

	particles.resize(num_particles);
	weights.resize(num_particles);

	double init_weight = 1.0 / num_particles;

	for(int i = 0; i < num_particles; i++)
	{
		particles[i].id = i;
		particles[i].x = d_x(rand_num);
		particles[i].y = d_y(rand_num);
		particles[i].theta = d_theta(rand_num);
		particles[i].weight = init_weight;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> d_x(0.0, std_pos[0]);
	normal_distribution<double> d_y(0.0, std_pos[1]);
	normal_distribution<double> d_theta(0.0, std_pos[2]);

	double v_dt = velocity * delta_t;
	double yaw_dt = yaw_rate * delta_t;
	double v_yaw = velocity / yaw_rate;

	for(int i = 0; i < num_particles; i++)
	{
		if(fabs(yaw_rate) < LOW_LIMIT)
		{
			particles[i].x += v_dt * cos(particles[i].theta);
			particles[i].y += v_dt * sin(particles[i].theta);
		}
		else
		{
			double new_theta = particles[i].theta + yaw_dt;
			particles[i].x += v_yaw * (sin(new_theta) - sin(particles[i].theta));
			particles[i].y += v_yaw * (cos(particles[i].theta) - cos(new_theta));
			particles[i].theta = new_theta;
		}
		particles[i].x += d_x(rand_num);
		particles[i].y += d_y(rand_num);
		particles[i].theta += d_theta(rand_num);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i = 0; i < observations.size(); i++)
	{
		LandmarkObs cur_observ = observations[i];
		double min_dist = numeric_limits<double>::max();
		int map_id = -1;
		for(unsigned int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs cur_pred = predicted[j];
			double cur_dist = dist(cur_observ.x, cur_observ.y, cur_pred.x, cur_pred.y);

			if(cur_dist < min_dist)
			{
				min_dist = cur_dist;
				map_id = cur_pred.id;
			}
		}
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	unsigned int landmark_size = map_landmarks.landmark_list.size();
	for(int i = 0; i < num_particles; i++)
	{
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> preds;
		
		for(unsigned int j = 0; j < landmark_size; j++)
		{
			double landmark_x = map_landmarks.landmark_list[j].x_f;
			double landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			if(fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range)
				preds.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
		}

		vector<LandmarkObs> trans;
		for(unsigned int j = 0; j < observations.size(); j++)
		{
			double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
			double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
			trans.push_back(LandmarkObs{observations[j].id, t_x, t_y});
		}

		dataAssociation(preds, trans);

		particles[i].weight = 1.0;
		for(unsigned int j = 0; j < trans.size(); j++)
		{
			double t_x_ = trans[j].x;
			double t_y_ = trans[j].y;
			int asso_pre_id = trans[j].id;

			double p_x_, p_y_;
			for(unsigned int k = 0; k < preds.size(); k++)
			{
				if(preds[k].id == asso_pre_id)
				{
					p_x_ = preds[k].x;
					p_y_ = preds[k].y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double tmp_ = exp(-(pow(p_x_-t_x_, 2)/(2*pow(std_x, 2)) + (pow(p_y_-t_y_, 2)/(2*pow(std_y, 2)))));
			double obsv_weight = (1 / (2*M_PI*std_x*std_y)) * tmp_;

			particles[i].weight *= obsv_weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	vector<double> weights;
	for(int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}
	uniform_int_distribution<int> uni_int_distri(0, num_particles - 1);
	auto index = uni_int_distri(rand_num);
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> uni_real_distri(0.0, max_weight);

	double beta = 0.0;
	for(int i = 0; i < num_particles; i++)
	{
		beta += uni_real_distri(rand_num) * 2.0;
		while(beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
