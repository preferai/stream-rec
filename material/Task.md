# Task

Let's imagine the following situation:
You become the CTO of a large streaming service, like Twitch, and during the onboarding you find out that they don't use recommendation systems or any kind of personalization in their product. Describe an implementation plan to integrate a recommendation system into the product to boost business KPIs like watch time, session length, ad revenue, etc.

Let's call the endpoints of the recommendation system "scenarios". Describe the list of scenarios which will need to be implemented, what kind of data will be sent in the requests to the endpoint, what kind of data will be necessary for training the underlying models of the endpoint and what will the endpoint return.

 

For example:

 

### Scenario: HOMETOWN

    Description: Order the streams of front page of the website so that streamers who are from the same city as the user are featured on top
    Request data: user id
    Returned data: list of streams from the same location
    Data used for training: user locations, streamer locations

 

### Scenario: SMS_ALERT

    Description: Send an SMS reminder to users who haven't accessed the website in >30 days to check their favorite streamers
    Request data: none
    Returned data: list of users who haven't accessed the website recently along with one of their bookmarked streamers
    Data used for training: streams bookmarked by the user, timestamp of streams, timestamp of user visits