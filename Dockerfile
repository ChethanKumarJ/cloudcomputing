# Use an official OpenJDK runtime as a parent image
FROM openjdk:8-jre-slim

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:${PATH}"

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y curl bzip2 wget --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Download and set up Spark
WORKDIR /opt
RUN wget --quiet -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" && \
    tar -xzf apache-spark.tgz && \
    rm apache-spark.tgz && \
    ln -s spark-3.5.0-bin-hadoop3 spark

# Add AWS SDK JARs for S3 integration
RUN wget -q -O /opt/spark/jars/aws-java-sdk-1.8.0.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.8.0/aws-java-sdk-1.8.0.jar && \
    wget -q -O /opt/spark/jars/hadoop-aws-3.0.0.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.0/hadoop-aws-3.0.0.jar

# Copy application files into the container
COPY WineQualityPrediction.java /opt/
COPY ValidationDataset.csv /opt/
COPY spark-model /opt/spark-model/

# Compile Java application
WORKDIR /opt
RUN apt-get update && apt-get install -y default-jdk && \
    javac -cp "$SPARK_HOME/jars/*" WineQualityPrediction.java

# Set the entry point and default command
CMD ["java", "-cp", "/opt/spark/jars/*:/opt", "WineQualityPrediction"]
