# Use an official Python runtime as a parent image
FROM openjdk:8-jre-slim

# Set environment variables for Miniconda
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"
ENV SPARK_HOME=/opt/spark

# Update and install necessary packages, install Miniconda
RUN apt-get update && \
    apt-get install -y curl bzip2 wget --no-install-recommends && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda3 && \
    rm -rf /var/lib/apt/lists/* /tmp/miniconda.sh

# Configure Miniconda
RUN conda config --set auto_update_conda false && \
    conda config --set show_channel_urls true

# Install Python packages
RUN conda install --yes --freeze-installed \
    numpy pandas && \
    pip install --no-cache-dir pyspark==3.5.0 awscli && \
    conda clean -afy

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
COPY prediction.py /opt/
COPY ValidationDataset.csv /opt/
COPY spark-model /opt/spark-model/

# Set the entry point and default command
CMD ["spark-submit", "/opt/prediction.py", "/opt/ValidationDataset.csv"]
