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

using namespace std;

static int NUM_PARTICLES = 100;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//normal distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
  	normal_distribution<double> dist_y(y, std[1]);
  	normal_distribution<double> dist_theta(theta, std[2]);
  	//default_random_engine gen;

	num_particles = NUM_PARTICLES;
	particles.resize(num_particles);
	weights.resize(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// normal distributions for sensor noise
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate)<0.0001){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i =0; i < observations.size(); i++){
		LandmarkObs obs = observations[i];

		double min_dist = numeric_limits<double>::max();

		int map_id = -1;

		for (int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred = predicted[j];

			double current_dist = dist(obs.x, obs.y, pred.x, pred.y);
			if (current_dist < min_dist){
				min_dist = current_dist;
				map_id = pred.id;
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

	for (int i = 0; i< num_particles; i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		vector<LandmarkObs> predictions;
		for (int j=0; j<map_landmarks.landmark_list.size(); j++){
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			if (fabs(landmark_x - p_x) <= sensor_range && fabs(landmark_y - p_y) <= sensor_range){
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y});
			}
		}

		vector<LandmarkObs> transformed_os;
		for (int j = 0; j < observations.size(); j++) {
      		double trans_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      		double trans_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      		transformed_os.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
  		}

		dataAssociation(predictions, transformed_os);

		particles[i].weight = 1.0;

		for (int j = 0; j < transformed_os.size(); j++) {
			double o_x, o_y, pr_x, pr_y;
			o_x = transformed_os[j].x;
			o_y = transformed_os[j].y;
			int associated_prediction = transformed_os[j].id;

			for (int k = 0; k < predictions.size(); k++)
			{
				if (predictions[k].id == associated_prediction)
				{
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double obs_weight = (1 / (2 * M_PI * std_x * std_y)) * exp(-(pow(pr_x - o_x, 2) / (2 * pow(std_x, 2)) + (pow(pr_y - o_y, 2) / (2 * pow(std_y, 2)))));

			particles[i].weight *= obs_weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	/*
	vector<Particle> resampled_particles;

	vector<double> weights;

	for (int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}
	*/
	vector<Particle> resampled_particles;

	vector<double> weights;
	
	for (int x = 0; x < num_particles; x++)
	{
		weights.push_back(particles[x].weight);
	}

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);

	double max_weight = *max_element(weights.begin(), weights.end());

	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// Spin the resample wheel
	for (int x = 0; x < num_particles; x++)
	{
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;
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
