from flask import Flask, request, jsonify
from langchain_groq import ChatGroq

app = Flask(__name__)

@app.route('/process_biodata', methods=['POST'])
def process_biodata():
    try:
        # Get user biodata from request
        user_data = request.json
        
        # Initialize the ChatGroq LLM
        llm = ChatGroq(
            api_key="groq_api_key_here",  # Replace with your actual Groq API key
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1000,
            max_retries=2
        )
        
        # Format the user data into a system message
        system_message = "You are a helpful fitness assistant that provides personalized recommendations based on user biodata."
        
        # Format the user data into a user message
        user_message = f"""
        User Biodata:
        Age: {user_data.get('age')}
        Height: {user_data.get('height')}
        Weight: {user_data.get('weight')}
        Gender: {user_data.get('gender')}
        Activity Level: {user_data.get('activity_level')}
        Health Goals: {user_data.get('health_goals')}
        #limitations or barriers
        #what do u want to achieve -> eg: gaining muscle, bulking, tone..
        # country
        # available equipment
        # days to workout
        # 
        
        Based on this information, provide personalized fitness and health recommendations.
        """
        
        # Create messages list for LangChain format
        messages = [
            ("system", system_message),
            ("human", user_message)
        ]
        
        # Call the LLM with the messages
        ai_response = llm.invoke(messages)
        
        # Return the response
        return jsonify({"recommendations": ai_response.content})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
